"""Estrategias de scalping intradía evaluadas sobre barras de 1 minuto del subyacente.

Cada estrategia recibe el contexto de un día (DayContext) y devuelve la PRIMERA señal
del día (o None). Solo una señal por día a propósito: con una cuenta menor a $25k la
regla PDT limita las operaciones intradía, así que hay que ser selectivo.

Una señal se evalúa al CIERRE de la barra de señal; la entrada se simula en la
apertura de la barra siguiente para no hacer trampa con información del futuro.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Callable

import numpy as np
import pandas as pd

MARKET_OPEN = time(9, 30)


def _t(value: str | time) -> time:
    return value if isinstance(value, time) else time.fromisoformat(value)


# --------------------------------------------------------------------------- contexto

@dataclass
class DayContext:
    day: date
    bars: pd.DataFrame          # sesión regular (9:30-16:00 ET), barras de 1 minuto
    premarket: pd.DataFrame     # 4:00-9:29 ET
    prev_close: float
    prev_high: float
    prev_low: float
    o: np.ndarray = field(init=False, repr=False)
    h: np.ndarray = field(init=False, repr=False)
    l: np.ndarray = field(init=False, repr=False)
    c: np.ndarray = field(init=False, repr=False)
    v: np.ndarray = field(init=False, repr=False)
    times: list[time] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.o = self.bars["open"].to_numpy(float)
        self.h = self.bars["high"].to_numpy(float)
        self.l = self.bars["low"].to_numpy(float)
        self.c = self.bars["close"].to_numpy(float)
        self.v = self.bars["volume"].to_numpy(float)
        self.times = [ts.time() for ts in self.bars.index]

    @property
    def open_price(self) -> float:
        return float(self.o[0])

    @property
    def gap_pct(self) -> float:
        return (self.open_price - self.prev_close) / self.prev_close * 100

    def first_pos_at_or_after(self, t: time) -> int:
        for i, bt in enumerate(self.times):
            if bt >= t:
                return i
        return len(self.times)

    def vwap(self) -> np.ndarray:
        typical = (self.h + self.l + self.c) / 3
        cum_vol = np.cumsum(self.v)
        with np.errstate(divide="ignore", invalid="ignore"):
            vw = np.cumsum(typical * self.v) / cum_vol
        return np.where(cum_vol > 0, vw, typical)


def ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def opening_range(ctx: DayContext, minutes: int) -> tuple[float, float, int] | None:
    """(máximo, mínimo, posición donde termina el rango) de los primeros `minutes` minutos."""
    end = (datetime.combine(ctx.day, MARKET_OPEN) + pd.Timedelta(minutes=minutes)).time()
    end_pos = ctx.first_pos_at_or_after(end)
    if end_pos < max(1, int(minutes * 0.6)) or end_pos >= len(ctx.times):
        return None
    return float(ctx.h[:end_pos].max()), float(ctx.l[:end_pos].min()), end_pos


def allowed_directions(gap_pct: float, direction_filter: str) -> set[int]:
    if direction_filter == "both":
        return {1, -1}
    gap_dir = 1 if gap_pct > 0 else -1 if gap_pct < 0 else 0
    if gap_dir == 0:
        return set()
    if direction_filter == "with_gap":
        return {gap_dir}
    if direction_filter == "against_gap":
        return {-gap_dir}
    raise ValueError(f"direction_filter desconocido: {direction_filter}")


# --------------------------------------------------------------------------- señales

@dataclass
class Signal:
    pos: int                 # posición de la barra de señal
    direction: int           # 1 = CALL (alcista), -1 = PUT (bajista)
    stop: float              # stop en precio del subyacente
    target_r: float | None = None      # objetivo en múltiplos del riesgo...
    target_price: float | None = None  # ...o un precio fijo
    exit_by: time = time(11, 30)       # salida por tiempo (scalping, no se queda todo el día)
    note: str = ""


def orb(ctx: DayContext, range_minutes: int = 5, target_r: float = 2.0, stop_mode: str = "range",
        direction_filter: str = "both", last_entry: str = "10:30", exit_by: str = "11:30") -> Signal | None:
    """Ruptura del rango de apertura (Opening Range Breakout)."""
    rng = opening_range(ctx, range_minutes)
    if rng is None:
        return None
    hi, lo, start = rng
    allowed = allowed_directions(ctx.gap_pct, direction_filter)
    mid = (hi + lo) / 2
    last = _t(last_entry)
    for i in range(start, len(ctx.c) - 1):
        if ctx.times[i] > last:
            break
        if ctx.c[i] > hi and 1 in allowed:
            return Signal(i, 1, lo if stop_mode == "range" else mid, target_r=target_r, exit_by=_t(exit_by),
                          note=f"Rompe máximo del rango {range_minutes}m ({hi:.2f})")
        if ctx.c[i] < lo and -1 in allowed:
            return Signal(i, -1, hi if stop_mode == "range" else mid, target_r=target_r, exit_by=_t(exit_by),
                          note=f"Rompe mínimo del rango {range_minutes}m ({lo:.2f})")
    return None


def premarket_break(ctx: DayContext, target_r: float = 2.0, stop_lookback: int = 5,
                    direction_filter: str = "both", last_entry: str = "10:30",
                    exit_by: str = "11:30") -> Signal | None:
    """Cruce del máximo o mínimo del pre-market después de la apertura."""
    if len(ctx.premarket) < 10:
        return None
    pmh = float(ctx.premarket["high"].max())
    pml = float(ctx.premarket["low"].min())
    allowed = allowed_directions(ctx.gap_pct, direction_filter)
    last = _t(last_entry)
    for i in range(1, len(ctx.c) - 1):
        if ctx.times[i] > last:
            break
        lb = max(0, i - stop_lookback + 1)
        if ctx.c[i - 1] <= pmh < ctx.c[i] and 1 in allowed:
            return Signal(i, 1, float(ctx.l[lb:i + 1].min()), target_r=target_r, exit_by=_t(exit_by),
                          note=f"Cruza máximo pre-market ({pmh:.2f})")
        if ctx.c[i - 1] >= pml > ctx.c[i] and -1 in allowed:
            return Signal(i, -1, float(ctx.h[lb:i + 1].max()), target_r=target_r, exit_by=_t(exit_by),
                          note=f"Cruza mínimo pre-market ({pml:.2f})")
    return None


def vwap_cross(ctx: DayContext, target_r: float = 1.5, stop_lookback: int = 5, ema_fast: int = 9,
               ema_slow: int = 20, start: str = "09:45", last_entry: str = "11:00",
               direction_filter: str = "both", exit_by: str = "11:45") -> Signal | None:
    """Recupera (o pierde) el VWAP con las EMAs a favor."""
    vw = ctx.vwap()
    ef, es = ema(ctx.c, ema_fast), ema(ctx.c, ema_slow)
    allowed = allowed_directions(ctx.gap_pct, direction_filter)
    last = _t(last_entry)
    for i in range(max(1, ctx.first_pos_at_or_after(_t(start))), len(ctx.c) - 1):
        if ctx.times[i] > last:
            break
        lb = max(0, i - stop_lookback + 1)
        if ctx.c[i - 1] <= vw[i - 1] and ctx.c[i] > vw[i] and ef[i] > es[i] and 1 in allowed:
            return Signal(i, 1, float(ctx.l[lb:i + 1].min()), target_r=target_r, exit_by=_t(exit_by),
                          note=f"Recupera VWAP ({vw[i]:.2f})")
        if ctx.c[i - 1] >= vw[i - 1] and ctx.c[i] < vw[i] and ef[i] < es[i] and -1 in allowed:
            return Signal(i, -1, float(ctx.h[lb:i + 1].max()), target_r=target_r, exit_by=_t(exit_by),
                          note=f"Pierde VWAP ({vw[i]:.2f})")
    return None


def gap_fade(ctx: DayContext, min_gap_pct: float = 0.3, max_gap_pct: float = 1.5, range_minutes: int = 5,
             fill_fraction: float = 1.0, last_entry: str = "10:30", exit_by: str = "11:30") -> Signal | None:
    """Apuesta a que el gap se rellena: entra en contra del gap cuando rompe el rango de apertura."""
    gap = ctx.gap_pct
    if not (min_gap_pct <= abs(gap) <= max_gap_pct):
        return None
    rng = opening_range(ctx, range_minutes)
    if rng is None:
        return None
    hi, lo, start = rng
    direction = -1 if gap > 0 else 1
    target = ctx.open_price - fill_fraction * (ctx.open_price - ctx.prev_close)
    last = _t(last_entry)
    for i in range(start, len(ctx.c) - 1):
        if ctx.times[i] > last:
            break
        if direction == -1 and ctx.c[i] < lo:
            return Signal(i, -1, hi, target_price=target, exit_by=_t(exit_by),
                          note=f"Gap arriba {gap:+.2f}%: busca rellenar hacia {target:.2f}")
        if direction == 1 and ctx.c[i] > hi:
            return Signal(i, 1, lo, target_price=target, exit_by=_t(exit_by),
                          note=f"Gap abajo {gap:+.2f}%: busca rellenar hacia {target:.2f}")
    return None


@dataclass(frozen=True)
class StrategySpec:
    fn: Callable[..., Signal | None]
    description: str
    grid: dict[str, list]


STRATEGIES: dict[str, StrategySpec] = {
    "orb": StrategySpec(orb, "Ruptura del rango de apertura", {
        "range_minutes": [5, 15],
        "target_r": [1.0, 1.5, 2.0],
        "stop_mode": ["range", "mid"],
        "direction_filter": ["both", "with_gap"],
    }),
    "premarket_break": StrategySpec(premarket_break, "Ruptura de máximo/mínimo pre-market", {
        "target_r": [1.0, 1.5, 2.0],
        "stop_lookback": [3, 5],
        "direction_filter": ["both", "with_gap"],
    }),
    "vwap_cross": StrategySpec(vwap_cross, "Recuperar/perder VWAP con EMAs a favor", {
        "target_r": [1.0, 1.5, 2.0],
        "stop_lookback": [3, 5],
        "direction_filter": ["both", "with_gap"],
    }),
    "gap_fade": StrategySpec(gap_fade, "Relleno de gap", {
        "min_gap_pct": [0.2, 0.4],
        "range_minutes": [5, 15],
        "fill_fraction": [0.5, 1.0],
    }),
}
