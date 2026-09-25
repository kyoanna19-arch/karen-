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
    live: bool = False          # en vivo se puede señalar en la última barra cerrada
    o: np.ndarray = field(init=False, repr=False)
    h: np.ndarray = field(init=False, repr=False)
    l: np.ndarray = field(init=False, repr=False)
    c: np.ndarray = field(init=False, repr=False)
    v: np.ndarray = field(init=False, repr=False)
    times: list[time] = field(init=False, repr=False)
    cache: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.o = self.bars["open"].to_numpy(float)
        self.h = self.bars["high"].to_numpy(float)
        self.l = self.bars["low"].to_numpy(float)
        self.c = self.bars["close"].to_numpy(float)
        self.v = self.bars["volume"].to_numpy(float)
        self.times = [ts.time() for ts in self.bars.index]

    @property
    def signal_end(self) -> int:
        """Límite (exclusivo) de barras donde puede haber señal. En backtest hace falta una barra
        siguiente para entrar; en vivo la última barra cerrada ya vale."""
        return len(self.c) if self.live else len(self.c) - 1

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
    for i in range(start, ctx.signal_end):
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
    for i in range(1, ctx.signal_end):
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
    for i in range(max(1, ctx.first_pos_at_or_after(_t(start))), ctx.signal_end):
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
    for i in range(start, ctx.signal_end):
        if ctx.times[i] > last:
            break
        if direction == -1 and ctx.c[i] < lo:
            return Signal(i, -1, hi, target_price=target, exit_by=_t(exit_by),
                          note=f"Gap arriba {gap:+.2f}%: busca rellenar hacia {target:.2f}")
        if direction == 1 and ctx.c[i] > hi:
            return Signal(i, 1, lo, target_price=target, exit_by=_t(exit_by),
                          note=f"Gap abajo {gap:+.2f}%: busca rellenar hacia {target:.2f}")
    return None


def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI de Wilder (el mismo que usan TC2000 y la mayoría de plataformas)."""
    delta = np.diff(values, prepend=values[0])
    gain = pd.Series(np.clip(delta, 0, None)).ewm(alpha=1 / period, adjust=False).mean()
    loss = pd.Series(np.clip(-delta, 0, None)).ewm(alpha=1 / period, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + gain / loss)
    return out.fillna(50).to_numpy()


def candles(ctx: DayContext, minutes: int) -> dict[str, np.ndarray]:
    """Velas de `minutes` minutos (pre-market + sesión regular) con EMA 9/21, RSI 14 y VWAP.

    El pre-market sirve para 'calentar' las EMAs y el RSI, como cuando ves la gráfica con
    horario extendido. El VWAP se reinicia a las 9:30. `end_pos` es la última barra de 1 minuto
    (en la sesión regular) de cada vela, que es cuando la vela cierra y se puede actuar."""
    key = ("candles", minutes)
    if key in ctx.cache:
        return ctx.cache[key]
    all_bars = pd.concat([ctx.premarket, ctx.bars])
    n_pre = len(ctx.premarket)
    ts = all_bars.index
    minute = (ts.hour * 60 + ts.minute).to_numpy()
    bucket = (minute - 570) // minutes  # 570 = 9:30; el pre-market queda en buckets negativos
    frame = pd.DataFrame({
        "open": all_bars["open"].to_numpy(float), "high": all_bars["high"].to_numpy(float),
        "low": all_bars["low"].to_numpy(float), "close": all_bars["close"].to_numpy(float),
        "volume": all_bars["volume"].to_numpy(float), "pos": np.arange(len(all_bars)) - n_pre,
        "minute": minute, "bucket": bucket,
    })
    g = frame.groupby("bucket", sort=True)
    agg = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(),
                        "end_pos": g["pos"].last(), "last_minute": g["minute"].last()})
    c = agg["close"].to_numpy()
    regular = agg.index.to_numpy() >= 0
    typical = ((agg["high"] + agg["low"] + agg["close"]) / 3).to_numpy()
    vol = np.where(regular, agg["volume"].to_numpy(), 0.0)
    cum_vol = np.cumsum(vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        vw = np.where(cum_vol > 0, np.cumsum(typical * vol) / cum_vol, np.nan)
    # una vela está completa si llegó su último minuto (o si ya hay velas después)
    expected_last = 570 + agg.index.to_numpy() * minutes + minutes - 1
    complete = agg["last_minute"].to_numpy() >= expected_last
    complete[:-1] = True
    out = {
        "open": agg["open"].to_numpy(), "high": agg["high"].to_numpy(), "low": agg["low"].to_numpy(),
        "close": c, "volume": agg["volume"].to_numpy(), "end_pos": agg["end_pos"].to_numpy(),
        "regular": regular, "complete": complete, "vwap": vw,
        "ema9": ema(c, 9), "ema21": ema(c, 21), "rsi": rsi(c, 14),
    }
    ctx.cache[key] = out
    return out


def ema_vwap_rsi(ctx: DayContext, timeframe: int = 5, target_r: float = 1.5, stop_mode: str = "candle",
                 body_min: float = 0.6, rsi_call: float = 60, rsi_put: float = 40, vol_mode: str = "prev",
                 vwap_slope_bars: int = 3, min_vwap_dist_pct: float = 0.0, max_vwap_dist_pct: float = 100.0,
                 start: str = "09:35", last_entry: str = "11:30", exit_by: str = "12:00") -> Signal | None:
    """Estrategia de Karen: EMA 9/21 + vela sólida que rompe los promedios + VWAP girado + RSI + volumen.

    CALL (PUT es el espejo):
      1. EMA 9 sobre EMA 21 y ambas apuntando hacia arriba
      2. Vela sólida alcista (cuerpo >= body_min del rango) que toca la EMA 9 y cierra sobre ambas EMAs
      3. Precio sobre el VWAP, VWAP subiendo en las últimas `vwap_slope_bars` velas,
         distancia al VWAP entre min_vwap_dist_pct y max_vwap_dist_pct (%)
      4. RSI(14) > rsi_call
      5. Volumen > vela anterior ('prev'), > las 4 anteriores ('max4') o > su promedio ('avg4')
    """
    k = candles(ctx, timeframe)
    o, h, l, c, v = k["open"], k["high"], k["low"], k["close"], k["volume"]
    e9, e21, r, vw = k["ema9"], k["ema21"], k["rsi"], k["vwap"]
    first, last = _t(start), _t(last_entry)
    for j in range(max(4, vwap_slope_bars), len(c)):
        pos = int(k["end_pos"][j])
        if not k["regular"][j] or not k["complete"][j] or pos >= ctx.signal_end:
            continue
        if ctx.times[pos] < first:
            continue
        if ctx.times[pos] > last:
            break
        rng = h[j] - l[j]
        if rng <= 0 or abs(c[j] - o[j]) / rng < body_min:
            continue
        if vol_mode == "prev":
            vol_ok = v[j] > v[j - 1]
        elif vol_mode == "max4":
            vol_ok = v[j] > v[j - 4:j].max()
        elif vol_mode == "avg4":
            vol_ok = v[j] > v[j - 4:j].mean()
        else:
            vol_ok = True
        if not vol_ok or np.isnan(vw[j]):
            continue
        dist = (c[j] - vw[j]) / vw[j] * 100
        prev_vw = vw[j - vwap_slope_bars] if vwap_slope_bars else vw[j]
        prev_vw = vw[j] if np.isnan(prev_vw) else prev_vw

        call = (e9[j] > e21[j] and e9[j] > e9[j - 1] and e21[j] > e21[j - 1]
                and c[j] > o[j] and l[j] <= e9[j] and c[j] > max(e9[j], e21[j])
                and min_vwap_dist_pct <= dist <= max_vwap_dist_pct
                and (vwap_slope_bars == 0 or vw[j] > prev_vw)
                and r[j] > rsi_call)
        put = (e9[j] < e21[j] and e9[j] < e9[j - 1] and e21[j] < e21[j - 1]
               and c[j] < o[j] and h[j] >= e9[j] and c[j] < min(e9[j], e21[j])
               and min_vwap_dist_pct <= -dist <= max_vwap_dist_pct
               and (vwap_slope_bars == 0 or vw[j] < prev_vw)
               and r[j] < rsi_put)
        if not (call or put):
            continue
        d = 1 if call else -1
        if stop_mode == "candle":
            stop = l[j] if d == 1 else h[j]
        else:  # "ema21"
            stop = e21[j]
        return Signal(pos, d, float(stop), target_r=target_r, exit_by=_t(exit_by),
                      note=f"Vela {timeframe}m sólida rompe EMAs | RSI {r[j]:.0f} | {dist:+.2f}% del VWAP")
    return None


@dataclass(frozen=True)
class StrategySpec:
    fn: Callable[..., Signal | None]
    description: str
    grid: dict[str, list]
    # variantes que apagan una regla a la vez, para ver si esa regla realmente ayuda
    ablations: dict[str, dict] = field(default_factory=dict)


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
    "ema_vwap_rsi": StrategySpec(ema_vwap_rsi, "Estrategia de Karen: EMA 9/21 + VWAP + RSI + volumen", {
        "timeframe": [1, 2, 5],
        "target_r": [1.0, 1.5, 2.0],
        "stop_mode": ["candle", "ema21"],
        "vol_mode": ["prev", "max4"],
        "max_vwap_dist_pct": [0.3, 100.0],
    }, ablations={
        "sin RSI": {"rsi_call": 0, "rsi_put": 100},
        "sin volumen": {"vol_mode": "none"},
        "sin vela sólida": {"body_min": 0.0},
        "sin VWAP girado": {"vwap_slope_bars": 0},
        "RSI 55/45": {"rsi_call": 55, "rsi_put": 45},
        "RSI 70/30": {"rsi_call": 70, "rsi_put": 30},
        "solo hasta 10:30": {"last_entry": "10:30"},
    }),
}
