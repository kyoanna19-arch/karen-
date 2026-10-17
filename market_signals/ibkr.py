"""Conexión con Interactive Brokers (TWS o IB Gateway) usando ib_async. SOLO cuentas paper.

Puertos por defecto de IBKR:
  TWS paper 7497 | IB Gateway paper 4002   (los puertos 7496/4001 son de cuentas reales)
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from .options import choose_expiration

LIVE_PORTS = {7496, 4001}
PENNY_SYMBOLS = {"SPY", "QQQ", "IWM"}  # opciones que cotizan en centavos en todos los precios


class IBKRError(RuntimeError):
    pass


def round_tick(price: float, symbol: str) -> float:
    """Las opciones de la mayoría de acciones cotizan en $0.05 por arriba de $3."""
    if symbol in PENNY_SYMBOLS or price < 3:
        return round(price, 2)
    return round(round(price / 0.05) * 0.05, 2)


def check_paper(accounts: list[str], port: int) -> None:
    if port in LIVE_PORTS:
        raise IBKRError(f"El puerto {port} es de una cuenta REAL. Usa 7497 (TWS paper) o 4002 (Gateway paper).")
    if not accounts or not all(a.upper().startswith("DU") for a in accounts):
        raise IBKRError(f"Las cuentas conectadas {accounts} no son paper (deben empezar con 'DU'). "
                        "Inicia sesión en TWS/Gateway en modo Paper Trading.")


def bars_to_frame(bars) -> pd.DataFrame:
    """Convierte barras de ib_async (fecha en UTC con formatDate=2) a hora del Este."""
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    idx = pd.DatetimeIndex([pd.Timestamp(b.date) for b in bars])
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York").tz_localize(None)
    df = pd.DataFrame({"open": [b.open for b in bars], "high": [b.high for b in bars],
                       "low": [b.low for b in bars], "close": [b.close for b in bars],
                       "volume": [float(b.volume) for b in bars]}, index=idx)
    df.index.name = "datetime"
    return df


class IBKRBroker:
    """Implementa tanto el feed (datos) como el broker (órdenes) de paper.py."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 7,
                 fill_timeout: float = 20):
        from ib_async import IB  # import aquí para que el resto del sistema funcione sin ib_async
        self.ib = IB()
        self.port = port
        self.fill_timeout = fill_timeout
        try:
            self.ib.connect(host, port, clientId=client_id, timeout=10)
        except Exception as exc:  # noqa: BLE001 - ib_async lanza varios tipos
            raise IBKRError(f"No pude conectar con TWS/Gateway en {host}:{port}. ¿Está abierto y con la API "
                            f"activada? Detalle: {exc}") from exc
        self.accounts = self.ib.managedAccounts()
        try:
            check_paper(self.accounts, port)
        except IBKRError:
            self.ib.disconnect()
            raise
        self.ib.reqMarketDataType(3)  # usa datos en vivo si los tienes, y si no, con retraso
        self._stocks: dict = {}

    def disconnect(self) -> None:
        self.ib.disconnect()

    def sleep(self, seconds: float) -> None:
        self.ib.sleep(seconds)  # mantiene viva la conexión mientras espera

    def _stock(self, symbol: str):
        from ib_async import Stock
        if symbol not in self._stocks:
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            self._stocks[symbol] = contract
        return self._stocks[symbol]

    def summary(self) -> str:
        values = {v.tag: v.value for v in self.ib.accountSummary() if v.currency in ("USD", "")}
        return (f"Cuenta(s): {', '.join(self.accounts)} | Liquidez neta: ${float(values.get('NetLiquidation', 0)):,.2f}"
                f" | Poder de compra: ${float(values.get('BuyingPower', 0)):,.2f}")

    # ------------------------------------------------------------------ datos (Feed)
    def minute_bars(self, symbol: str, now: datetime) -> pd.DataFrame:
        bars = self.ib.reqHistoricalData(self._stock(symbol), endDateTime="", durationStr="1 D",
                                         barSizeSetting="1 min", whatToShow="TRADES", useRTH=False,
                                         formatDate=2)
        df = bars_to_frame(bars)
        return df[df.index.date == now.date()]

    def prev_day(self, symbol: str, today: date) -> dict:
        bars = self.ib.reqHistoricalData(self._stock(symbol), endDateTime="", durationStr="10 D",
                                         barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1)
        prior = [b for b in bars if pd.Timestamp(b.date).date() < today]
        if not prior:
            raise IBKRError(f"No hay datos del día anterior para {symbol}")
        b = prior[-1]
        return {"close": b.close, "high": b.high, "low": b.low}

    def option_candidates(self, symbol: str, direction: int, price: float, today: date) -> list[dict]:
        from ib_async import Option
        stock = self._stock(symbol)
        chains = self.ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        chain = next((c for c in chains if c.exchange == "SMART" and c.tradingClass == symbol), None)
        if chain is None:
            return []
        expirations = [f"{e[:4]}-{e[4:6]}-{e[6:]}" for e in chain.expirations]
        exp = choose_expiration(expirations, today)
        if exp is None:
            return []
        strikes = sorted(chain.strikes, key=lambda s: abs(s - price))[:8]
        right = "C" if direction == 1 else "P"
        contracts = [Option(symbol, exp.replace("-", ""), s, right, "SMART", tradingClass=chain.tradingClass,
                            multiplier=chain.multiplier, currency="USD") for s in strikes]
        contracts = [c for c in self.ib.qualifyContracts(*contracts) if c and c.conId]
        out = []
        for t in self.ib.reqTickers(*contracts):
            greeks = t.modelGreeks
            out.append({"symbol": t.contract.localSymbol, "underlying": symbol, "_contract": t.contract,
                        "option_type": "call" if right == "C" else "put",
                        "strike": t.contract.strike, "expiration": exp,
                        "bid": t.bid if t.bid == t.bid and t.bid > 0 else 0,
                        "ask": t.ask if t.ask == t.ask and t.ask > 0 else 0,
                        "greeks": {"delta": greeks.delta} if greeks and greeks.delta is not None else None})
        return out

    # ------------------------------------------------------------------ órdenes (Broker)
    def _option(self, option: dict):
        from ib_async import Option
        if option.get("_contract") is not None:
            return option["_contract"]
        right = "C" if option["option_type"] == "call" else "P"
        contract = Option(option["underlying"], option["expiration"].replace("-", ""), float(option["strike"]), right,
                          "SMART", currency="USD", multiplier="100")
        qualified = self.ib.qualifyContracts(contract)
        if not qualified or not qualified[0] or not qualified[0].conId:
            raise IBKRError(f"IBKR no reconoce el contrato {option['symbol']}")
        return qualified[0]

    def _wait(self, trade) -> None:
        deadline = datetime.now() + timedelta(seconds=self.fill_timeout)
        while not trade.isDone() and datetime.now() < deadline:
            self.ib.sleep(0.5)

    def buy(self, option: dict, qty: int, limit: float) -> tuple[int, float] | None:
        from ib_async import LimitOrder
        contract = self._option(option)
        option["_contract"] = contract
        trade = self.ib.placeOrder(contract, LimitOrder("BUY", qty, round_tick(limit, contract.symbol), tif="DAY"))
        self._wait(trade)
        filled = int(trade.orderStatus.filled)
        if not trade.isDone():
            self.ib.cancelOrder(trade.order)
            self.ib.sleep(1)
            filled = int(trade.orderStatus.filled)
        if filled <= 0:
            return None
        return filled, float(trade.orderStatus.avgFillPrice)

    def protective_stop(self, option: dict, qty: int, stop_price: float):
        from ib_async import StopOrder
        contract = option["_contract"]
        return self.ib.placeOrder(contract, StopOrder("SELL", qty, round_tick(stop_price, contract.symbol),
                                                      tif="DAY"))

    def stop_fill(self, handle) -> float | None:
        if handle is not None and handle.orderStatus.status == "Filled":
            return float(handle.orderStatus.avgFillPrice)
        return None

    def sell(self, option: dict, qty: int, stop_handle) -> float:
        from ib_async import MarketOrder
        if stop_handle is not None and not stop_handle.isDone():
            self.ib.cancelOrder(stop_handle.order)
            self.ib.sleep(1)
            if stop_handle.orderStatus.status == "Filled":  # se llenó justo antes de cancelarlo
                return float(stop_handle.orderStatus.avgFillPrice)
        trade = self.ib.placeOrder(option["_contract"], MarketOrder("SELL", qty, tif="DAY"))
        self._wait(trade)
        if trade.orderStatus.status != "Filled":
            raise IBKRError(f"La venta de {option['symbol']} no se llenó. ¡Revisa la posición en TWS!")
        return float(trade.orderStatus.avgFillPrice)
