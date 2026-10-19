"""Paper trading automático: detecta la señal, compra la opción, la vigila y la cierra.

Separa dos piezas para poder mezclar proveedores:
  - feed:   de dónde salen las velas y la cadena de opciones (Tradier o IBKR)
  - broker: dónde se mandan las órdenes (IBKR paper)

Las salidas siguen las mismas reglas que el backtest (stop, objetivo y salida por tiempo
medidos en el SUBYACENTE), para que los resultados en paper se puedan comparar con él.
Además se deja un stop de protección sobre la prima por si el programa se cierra.
"""
from __future__ import annotations

import csv
import json
import time as systime
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from .config import Settings
from .monitor import live_signal
from .options import contracts_for_risk, pick_contract
from .options_sim import ExitPlan
from .strategies import STRATEGIES, DayContext, Signal

JOURNAL = Path("data/paper_journal.csv")


class Feed(Protocol):
    def minute_bars(self, symbol: str, now: datetime) -> pd.DataFrame: ...
    def prev_day(self, symbol: str, today: date) -> dict: ...
    def option_candidates(self, symbol: str, direction: int, price: float, today: date,
                          min_dte: int = 0) -> list[dict]: ...


class Broker(Protocol):
    def buy(self, option: dict, qty: int, limit: float) -> tuple[int, float] | None: ...
    def protective_stop(self, option: dict, qty: int, stop_price: float): ...
    def sell_limit(self, option: dict, qty: int, price: float): ...
    def order_fill(self, handle) -> float | None: ...
    def cancel(self, handle) -> None: ...
    def sell(self, option: dict, qty: int, open_orders: list) -> float: ...


def split_qty(qty: int, tiers) -> list[tuple[int, float]]:
    """Reparte los contratos entre los escalones: [(contratos, ganancia)].
    Con pocos contratos los primeros escalones tienen prioridad (con 1 contrato todo sale en el primero)."""
    out, left = [], qty
    for gain, frac in tiers:
        q = min(left, max(1, round(frac * qty))) if left > 0 else 0
        out.append((q, gain))
        left -= q
    if left > 0 and out and sum(f for _, f in tiers) >= 0.999:
        out[-1] = (out[-1][0] + left, out[-1][1])
    return out


@dataclass
class PaperTrade:
    date: str
    symbol: str
    strategy: str
    params: str
    side: str
    signal_time: str
    underlying_entry: float
    underlying_stop: float
    underlying_target: float
    option: str
    qty: int
    option_entry: float
    option_exit: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    pnl: float = 0.0
    r_underlying: float = 0.0


def append_journal(trade: PaperTrade, path: Path = JOURNAL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(trade)))
        if new:
            writer.writeheader()
        writer.writerow(asdict(trade))


def closed_bars(bars: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Solo velas de 1 minuto que ya cerraron."""
    return bars[bars.index + timedelta(minutes=1) <= now]


class PaperTrader:
    def __init__(self, feed: Feed, broker: Broker, settings: Settings, notify: Callable[[str], None],
                 symbol: str, strategy: str, params: dict, now: Callable[[], datetime],
                 sleep: Callable[[float], None] = systime.sleep, poll_seconds: float = 15,
                 until: str = "12:00", journal: Path = JOURNAL, exit_plan: ExitPlan | None = None,
                 min_dte: int = 0, target_delta: float = 0.45):
        self.feed, self.broker, self.settings, self.notify = feed, broker, settings, notify
        self.symbol, self.strategy, self.params = symbol, strategy, params
        self.now, self.sleep, self.poll_seconds = now, sleep, poll_seconds
        self.until = time.fromisoformat(until)
        self.journal = journal
        self.sig: Signal | None = None
        self.trade: PaperTrade | None = None
        self.option: dict | None = None
        self.stop_handle = None
        self.checked_pos = -1
        self.target = 0.0
        self.entry_price = 0.0
        self.plan = exit_plan
        self.min_dte, self.target_delta = min_dte, target_delta
        self.entry_time: datetime | None = None
        self.fills: list[tuple[int, float, str]] = []  # (contratos, precio, motivo)
        self.remaining = 0
        self.tier_orders: list[list] = []  # [orden, contratos, ganancia, llenada]
        self.at_breakeven = False

    # ------------------------------------------------------------------ ciclo principal
    def run(self) -> PaperTrade | None:
        today = self.now().date()
        prev = self.feed.prev_day(self.symbol, today)
        self.notify(f"🧪 PAPER: {self.symbol} con {self.strategy} {self.params or ''}. "
                    f"Vigilando hasta las {self.until:%H:%M} ET.")
        while True:
            now = self.now()
            bars = closed_bars(self.feed.minute_bars(self.symbol, now), now)
            regular = bars.between_time("09:30", "15:59")
            if self.trade is None:
                if now.time() >= self.until:
                    self.notify(f"PAPER {self.symbol}: sin señal hoy. No operar también es una decisión.")
                    return None
                if len(regular) >= 2:
                    ctx = DayContext(day=today, bars=regular, premarket=bars.between_time("04:00", "09:29"),
                                     prev_close=prev["close"], prev_high=prev["high"], prev_low=prev["low"])
                    sig = live_signal(ctx, self.strategy, self.params)
                    if sig is not None:
                        if sig.pos < len(ctx.c) - 3:
                            self.notify(f"PAPER {self.symbol}: la señal fue a las {ctx.times[sig.pos]:%H:%M} ET; "
                                        "ya es tarde para entrar.")
                            return None
                        if not self._enter(ctx, sig, today):
                            return None
            elif self._manage(regular, now):
                return self.trade
            self.sleep(self.poll_seconds)

    # ------------------------------------------------------------------ entrada
    def _enter(self, ctx: DayContext, sig: Signal, today: date) -> bool:
        price = float(ctx.c[sig.pos])
        risk = abs(price - sig.stop)
        if risk <= 0:
            return False
        self.target = sig.target_price if sig.target_price is not None else price + sig.direction * sig.target_r * risk
        side = "CALL" if sig.direction == 1 else "PUT"
        option = pick_contract(self.feed.option_candidates(self.symbol, sig.direction, price, today,
                                                           min_dte=self.min_dte),
                               sig.direction, price, target_delta=self.target_delta)
        if option is None:
            self.notify(f"PAPER {self.symbol}: señal {side} pero sin contrato con spread aceptable. No entro.")
            return False
        stop_pct = self.plan.stop_pct * 100 if self.plan else self.settings.option_stop_pct
        if self.plan:
            qty = contracts_for_risk(self.settings.account_size, self.settings.risk_per_trade_pct, option["ask"],
                                     stop_pct)
        else:
            # salida por niveles del ETF: se pierde aprox. delta x distancia al stop (+ el spread)
            delta = abs((option.get("greeks") or {}).get("delta") or self.target_delta)
            loss_per_contract = (delta * risk + max(option["ask"] - option.get("bid", 0), 0)) * 100
            risk_dollars = self.settings.account_size * self.settings.risk_per_trade_pct / 100
            qty = int(risk_dollars // loss_per_contract) if loss_per_contract > 0 else 0
        max_by_cash = int(self.settings.account_size * self.settings.max_position_pct / 100 // (option["ask"] * 100))
        qty = min(qty, max_by_cash)
        if qty == 0:
            self.notify(f"PAPER {self.symbol}: señal {side} pero la prima ({option['ask']}) es muy cara "
                        f"para tu riesgo por trade o pasa del {self.settings.max_position_pct:.0f}% de la cuenta. "
                        "No entro.")
            return False
        filled = self.broker.buy(option, qty, option["ask"])
        if filled is None:
            self.notify(f"PAPER {self.symbol}: la orden de compra de {option['symbol']} no se llenó. Cancelada.")
            return False
        qty, fill = filled
        self.remaining = qty
        self.entry_time = self.now()
        stop_price = fill * (1 - stop_pct / 100)
        self.stop_handle = self.broker.protective_stop(option, qty, stop_price)
        tiers_text = ""
        if self.plan:
            if self.plan.max_minutes:
                tiers_text += f"\nSi no funciona en {self.plan.max_minutes} min, se cierra"
            for q, gain in split_qty(qty, self.plan.tiers):
                if q > 0:
                    self.tier_orders.append([self.broker.sell_limit(option, q, fill * (1 + gain)), q, gain, False])
                    tiers_text += f"\nVenta límite: {q} a ${fill * (1 + gain):.2f} (+{int(gain * 100)}%)"
            if qty - sum(t[1] for t in self.tier_orders) > 0:
                tiers_text += f"\n{qty - sum(t[1] for t in self.tier_orders)} contrato(s) corren hasta la hora límite"
        self.sig, self.option, self.checked_pos, self.entry_price = sig, option, sig.pos, price
        self.trade = PaperTrade(
            date=str(today), symbol=self.symbol, strategy=self.strategy, params=json.dumps(self.params),
            side=side, signal_time=f"{ctx.times[sig.pos]:%H:%M}", underlying_entry=round(price, 2),
            underlying_stop=round(sig.stop, 2), underlying_target=round(self.target, 2),
            option=option["symbol"], qty=qty, option_entry=fill)
        levels = (f"Subyacente {price:.2f} | salida máx. {sig.exit_by:%H:%M} ET" if self.plan else
                  f"Subyacente {price:.2f} | stop {sig.stop:.2f} | objetivo {self.target:.2f} | "
                  f"salida máx. {sig.exit_by:%H:%M} ET")
        self.notify(
            f"🧪 PAPER ENTRADA {self.symbol} {side}\n{sig.note}\n"
            f"Compré {qty} x {option['symbol']} a ${fill:.2f}\n{levels}\n"
            f"Stop en la opción: ${stop_price:.2f} (-{stop_pct:.0f}%){tiers_text}")
        return True

    # ------------------------------------------------------------------ manejo de la posición
    def _open_orders(self) -> list:
        return [self.stop_handle] + [t[0] for t in self.tier_orders if not t[3]]

    def _manage(self, regular: pd.DataFrame, now: datetime) -> bool:
        sig = self.sig
        protective = self.broker.order_fill(self.stop_handle)
        if protective is not None:
            return self._close(protective, "breakeven" if self.at_breakeven else "stop opción", now,
                               already_sold=True)

        if self.plan:
            for tier in self.tier_orders:
                if tier[3]:
                    continue
                price = self.broker.order_fill(tier[0])
                if price is None:
                    continue
                tier[3] = True
                self.fills.append((tier[1], price, f"+{int(tier[2] * 100)}%"))
                self.remaining -= tier[1]
                self.notify(f"💰 PAPER {self.symbol}: vendí {tier[1]} a ${price:.2f} (+{int(tier[2] * 100)}%). "
                            f"Quedan {self.remaining}.")
                if self.remaining <= 0:
                    self.broker.cancel(self.stop_handle)
                    return self._finish(now)
                self.broker.cancel(self.stop_handle)  # el stop se ajusta a los contratos que quedan
                stop_price = (self.trade.option_entry if self.plan.breakeven
                              else self.trade.option_entry * (1 - self.plan.stop_pct))
                self.at_breakeven = self.at_breakeven or self.plan.breakeven
                self.stop_handle = self.broker.protective_stop(self.option, self.remaining, stop_price)
                if self.plan.breakeven:
                    self.notify(f"PAPER {self.symbol}: stop movido a breakeven (${stop_price:.2f}).")
            closed_after_limit = any(t.time() >= sig.exit_by for t in regular.index[self.checked_pos + 1:])
            if now.time() >= sig.exit_by or closed_after_limit:
                return self._close(None, "tiempo", now)
            if self.plan.max_minutes and now - self.entry_time >= timedelta(minutes=self.plan.max_minutes):
                return self._close(None, f"no funcionó en {self.plan.max_minutes} min", now)
            return False

        d = sig.direction
        times = list(regular.index)
        for i in range(self.checked_pos + 1, len(times)):
            hi, lo = float(regular["high"].iloc[i]), float(regular["low"].iloc[i])
            worst, best = (lo, hi) if d == 1 else (hi, lo)
            self.checked_pos = i
            if times[i].time() >= sig.exit_by:
                return self._close(None, "tiempo", now)
            if (worst - sig.stop) * d <= 0:
                return self._close(None, "stop", now, exit_underlying=sig.stop)
            if (best - self.target) * d >= 0:
                return self._close(None, "objetivo", now, exit_underlying=self.target)
        if now.time() >= sig.exit_by:
            return self._close(None, "tiempo", now)
        return False

    def _close(self, price: float | None, reason: str, now: datetime, already_sold: bool = False,
               exit_underlying: float | None = None) -> bool:
        """Cierra los contratos que quedan (vendiendo, o porque el stop ya los vendió)."""
        if self.remaining > 0:
            if already_sold:
                for tier in self.tier_orders:
                    if not tier[3]:
                        self.broker.cancel(tier[0])
            else:
                price = self.broker.sell(self.option, self.remaining, self._open_orders())
            self.fills.append((self.remaining, float(price), reason))
            self.remaining = 0
        risk = abs(self.entry_price - self.sig.stop)
        if exit_underlying is not None and risk > 0:
            self.trade.r_underlying = round((exit_underlying - self.entry_price) * self.sig.direction / risk, 2)
        return self._finish(now)

    def _finish(self, now: datetime) -> bool:
        trade = self.trade
        sold = sum(q for q, _, _ in self.fills)
        trade.option_exit = round(sum(q * p for q, p, _ in self.fills) / sold, 2)
        trade.exit_time = f"{now:%H:%M}"
        trade.exit_reason = " / ".join(r for _, _, r in self.fills)
        trade.pnl = round(sum((p - trade.option_entry) * 100 * q for q, p, _ in self.fills), 2)
        append_journal(trade, self.journal)
        emoji = "✅" if trade.pnl > 0 else "❌"
        detail = "\n".join(f"  {q} a ${p:.2f} ({r})" for q, p, r in self.fills)
        self.notify(f"{emoji} PAPER SALIDA {self.symbol} {trade.side}\n"
                    f"{trade.qty} x {trade.option}, entrada ${trade.option_entry:.2f}\n{detail}\n"
                    f"Resultado: ${trade.pnl:+,.2f}")
        return True


def journal_summary(path: Path = JOURNAL) -> str:
    if not path.exists():
        return "Todavía no hay trades en la bitácora de paper."
    df = pd.read_csv(path)
    if df.empty:
        return "Todavía no hay trades en la bitácora de paper."
    df["month"] = df["date"].str[:7]
    lines = [f"Trades: {len(df)} | Ganadores: {(df['pnl'] > 0).mean():.0%} | "
             f"Total: ${df['pnl'].sum():+,.2f} | Promedio: ${df['pnl'].mean():+,.2f}", "", "Por mes:"]
    for month, g in df.groupby("month"):
        lines.append(f"  {month}: {len(g)} trades, {(g['pnl'] > 0).mean():.0%} ganadores, ${g['pnl'].sum():+,.2f}")
    lines += ["", "Por motivo de salida:"]
    for reason, g in df.groupby("exit_reason"):
        lines.append(f"  {reason}: {len(g)} trades, ${g['pnl'].sum():+,.2f}")
    return "\n".join(lines)
