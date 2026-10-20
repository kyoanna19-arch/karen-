"""Portafolio de swing: varios activos compartiendo la misma cuenta, día por día.

El mismo motor (`step`) se usa para:
  - el backtest de muchos años (se llama una vez por cada día de la historia)
  - el escáner diario de paper/real (se llama solo con el día de hoy, guardando el estado en un archivo)
Así lo que se opera en paper es exactamente lo que se simuló.

Funcionamiento de cada día:
  1. En la APERTURA se ejecutan las órdenes decididas el día anterior (primero ventas, luego compras).
  2. Con el CIERRE se revisan las señales: salidas de lo que tenemos y entradas nuevas.
     Las entradas se ordenan por la caída más fuerte (RSI2 más bajo) y se limitan a los espacios libres.
     Cada espacio = capital total / max_positions. Se compran acciones ENTERAS.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .swing import SwingRule, cost_for, entry_signal, exit_signal, indicators, load_daily, daily_path


@dataclass
class Position:
    qty: float
    entry_date: str
    entry_price: float
    entry_cost: float  # dinero total pagado (precio + costos + comisión)


@dataclass
class Order:
    symbol: str
    side: str  # "BUY" o "SELL"
    qty: float
    signal_date: str


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    pending: list[Order] = field(default_factory=list)
    last_date: str | None = None
    closed: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"cash": self.cash, "last_date": self.last_date,
                           "positions": {k: asdict(v) for k, v in self.positions.items()},
                           "pending": [asdict(o) for o in self.pending], "closed": self.closed}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PortfolioState":
        d = json.loads(text)
        return cls(cash=d["cash"], last_date=d["last_date"],
                   positions={k: Position(**v) for k, v in d["positions"].items()},
                   pending=[Order(**o) for o in d["pending"]], closed=d.get("closed", []))

    def equity(self, ind: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        value = self.cash
        for sym, pos in self.positions.items():
            df = ind[sym]
            px = df.at[date, "close"] if date in df.index else df.loc[:date, "close"].iloc[-1]
            value += pos.qty * px
        return value


@dataclass(frozen=True)
class PortfolioConfig:
    rule: SwingRule = SwingRule()
    max_positions: int = 2
    commission: float = 1.0      # dólares por orden
    whole_shares: bool = True


def _ts(d: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d)


def step(state: PortfolioState, date: pd.Timestamp, ind: dict[str, pd.DataFrame], cfg: PortfolioConfig) -> dict:
    """Avanza un día. Devuelve lo que pasó: ejecuciones en la apertura y órdenes nuevas para mañana."""
    filled, new_orders = [], []
    # 1. Ejecutar en la apertura las órdenes del día anterior (ventas primero para liberar dinero)
    for order in sorted(list(state.pending), key=lambda o: o.side != "SELL"):
        df = ind[order.symbol]
        if date not in df.index or date <= _ts(order.signal_date):
            continue
        px = float(df.at[date, "open"])
        cost = cost_for(order.symbol)
        state.pending.remove(order)
        if order.side == "SELL":
            pos = state.positions.pop(order.symbol, None)
            if pos is None:
                continue
            proceeds = pos.qty * px * (1 - cost) - cfg.commission
            state.cash += proceeds
            trade = {"symbol": order.symbol, "entry_date": pos.entry_date, "exit_date": str(date.date()),
                     "qty": pos.qty, "entry_price": round(pos.entry_price, 2), "exit_price": round(px, 2),
                     "pnl": round(proceeds - pos.entry_cost, 2), "ret": round(proceeds / pos.entry_cost - 1, 5)}
            state.closed.append(trade)
            filled.append({"side": "SELL", **trade})
        else:
            qty = order.qty
            total = qty * px * (1 + cost) + cfg.commission
            if total > state.cash:  # abrió más caro: comprar lo que alcance
                qty = (state.cash - cfg.commission) / (px * (1 + cost))
                qty = math.floor(qty) if cfg.whole_shares else qty
                total = qty * px * (1 + cost) + cfg.commission
            if qty <= 0:
                continue
            state.cash -= total
            state.positions[order.symbol] = Position(qty, str(date.date()), px, total)
            filled.append({"side": "BUY", "symbol": order.symbol, "qty": qty, "price": round(px, 2)})

    # 2. Señales con el cierre
    exiting = []
    for sym, pos in state.positions.items():
        df = ind[sym]
        if date not in df.index:
            continue
        held = df.index.get_loc(date) - df.index.get_loc(_ts(pos.entry_date))
        if exit_signal(df.loc[date], cfg.rule) or held >= cfg.rule.max_days:
            order = Order(sym, "SELL", pos.qty, str(date.date()))
            state.pending.append(order)
            new_orders.append(order)
            exiting.append(sym)

    pending_buys = {o.symbol for o in state.pending if o.side == "BUY"}
    staying = len(state.positions) - len(exiting)
    slots = cfg.max_positions - staying - len(pending_buys)
    if slots > 0:
        candidates = []
        for sym, df in ind.items():
            if sym in state.positions or sym in pending_buys or date not in df.index:
                continue
            row = df.loc[date]
            if not pd.isna(row["sma200"]) and entry_signal(row, cfg.rule):
                candidates.append((row["rsi2"], sym, float(row["close"])))
        equity = state.equity(ind, date)
        slot_value = equity / cfg.max_positions
        for _, sym, close in sorted(candidates)[:slots]:
            qty = slot_value / (close * (1 + cost_for(sym)))
            qty = math.floor(qty) if cfg.whole_shares else qty
            if qty > 0:
                order = Order(sym, "BUY", qty, str(date.date()))
                state.pending.append(order)
                new_orders.append(order)
    state.last_date = str(date.date())
    return {"date": str(date.date()), "filled": filled, "orders": new_orders}


def load_indicators(symbols: list[str], folder: str | Path = "data") -> tuple[dict[str, pd.DataFrame], list[str]]:
    ind, missing = {}, []
    for sym in symbols:
        path = daily_path(sym, folder)
        if not path.exists():
            missing.append(sym)
            continue
        ind[sym.upper()] = indicators(load_daily(path))
    return ind, missing


def run_backtest(ind: dict[str, pd.DataFrame], cfg: PortfolioConfig, account: float = 3000,
                 start: str | None = None) -> tuple[PortfolioState, pd.Series]:
    dates = sorted(set().union(*[set(df.dropna(subset=["sma200"]).index) for df in ind.values()]))
    if start:
        dates = [d for d in dates if d >= pd.Timestamp(start)]
    state = PortfolioState(cash=account)
    equity = []
    for date in dates:
        step(state, date, ind, cfg)
        equity.append(state.equity(ind, date))
    return state, pd.Series(equity, index=pd.DatetimeIndex(dates))


def portfolio_stats(state: PortfolioState, equity: pd.Series, account: float,
                    benchmark: pd.DataFrame | None = None) -> dict:
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    rets = np.array([t["ret"] for t in state.closed])
    out = {
        "trades": len(rets), "trades_por_año": round(len(rets) / years, 1),
        "ganadores": round(float((rets > 0).mean()), 2) if len(rets) else 0.0,
        "prom_%": round(float(rets.mean()) * 100, 2) if len(rets) else 0.0,
        "final_$": round(float(equity.iloc[-1]), 0),
        "CAGR_%": round(((equity.iloc[-1] / account) ** (1 / years) - 1) * 100, 1),
        "caída_máx_%": round(float((equity / equity.cummax() - 1).min()) * 100, 1),
    }
    if benchmark is not None:
        bh = benchmark.loc[equity.index[0]:equity.index[-1], "close"]
        out["SPY_mantener_CAGR_%"] = round(((bh.iloc[-1] / bh.iloc[0]) ** (1 / years) - 1) * 100, 1)
        out["SPY_mantener_caída_%"] = round(float((bh / bh.cummax() - 1).min()) * 100, 1)
    return out


def yearly_returns(equity: pd.Series) -> pd.DataFrame:
    by_year = equity.groupby(equity.index.year).last()
    first = equity.iloc[0]
    prev = by_year.shift(1).fillna(first)
    return pd.DataFrame({"capital_fin_$": by_year.round(0), "resultado_%": ((by_year / prev - 1) * 100).round(1)})


# --------------------------------------------------------------------------- estado para el escáner diario

STATE_FILE = Path("data/swing_state.json")
JOURNAL_FILE = Path("data/swing_journal.csv")


def load_state(path: Path, account: float) -> PortfolioState:
    if path.exists():
        return PortfolioState.from_json(path.read_text(encoding="utf-8"))
    return PortfolioState(cash=account)


def save_state(state: PortfolioState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json(), encoding="utf-8")


def append_journal(trades: list[dict], path: Path) -> None:
    if not trades:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trades)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def scan(state: PortfolioState, ind: dict[str, pd.DataFrame], cfg: PortfolioConfig) -> list[dict]:
    """Avanza el estado desde el último día procesado hasta el último día con datos.
    La primera vez solo procesa el último día (no simula el pasado)."""
    all_dates = sorted(set().union(*[set(df.index) for df in ind.values()]))
    if state.last_date is None:
        dates = all_dates[-1:]
    else:
        dates = [d for d in all_dates if d > pd.Timestamp(state.last_date)]
    return [step(state, d, ind, cfg) for d in dates]
