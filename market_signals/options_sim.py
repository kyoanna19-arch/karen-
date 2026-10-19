"""Simulación del precio de la opción para probar stops sobre la prima y salidas escalonadas.

No tenemos precios históricos de opciones minuto a minuto (son muy caros), así que se estiman
con Black-Scholes a partir del precio de SPY en cada minuto. El modelo captura lo esencial
para scalping: delta, gamma y la pérdida por tiempo (theta), que en 0DTE es muy fuerte.
Es una aproximación: la volatilidad implícita real cambia durante el día. Por eso se puede
probar con distintas volatilidades (`iv`) para ver qué tan sensibles son los resultados.

Costos: `slip` es lo que se pierde por lado en dólares por acción (0.02 = 2 centavos):
medio spread + deslizamiento. Se paga al comprar y al vender.

Unidades: la ganancia de cada trade se expresa en R, donde 1R = el stop. Con un stop de 25%,
perder el 25% de la prima es -1R y ganar 50% es +2R.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import time

import numpy as np

from .backtest import Trade
from .strategies import DayContext, Signal

TRADING_MINUTES_YEAR = 252 * 390
CLOSE_MINUTE = 16 * 60


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2) * T) / v
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d1 - v)


def bs_put(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    return bs_call(S, K, T, sigma, r) - S + K * math.exp(-r * T)


def bs_delta(S: float, K: float, T: float, sigma: float, call: bool = True, r: float = 0.04) -> float:
    if T <= 0 or sigma <= 0:
        return (1.0 if S > K else 0.0) if call else (-1.0 if S < K else 0.0)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) if call else _ncdf(d1) - 1


def years_left(t: time, dte_days: int = 0) -> float:
    """Tiempo hasta el vencimiento en años de trading, contado desde el minuto `t`."""
    minutes = max(CLOSE_MINUTE - (t.hour * 60 + t.minute), 1) + dte_days * 390
    return minutes / TRADING_MINUTES_YEAR


def strike_for_delta(S: float, T: float, sigma: float, target: float = 0.45, call: bool = True,
                     step: float = 1.0) -> float:
    base = round(S / step) * step
    strikes = [base + k * step for k in range(-15, 16)]
    return min(strikes, key=lambda K: abs(abs(bs_delta(S, K, T, sigma, call)) - target))


@dataclass(frozen=True)
class ExitPlan:
    """stop_pct: pérdida máxima sobre la prima (0.25 = 25%).
    tiers: [(ganancia, fracción)], ej. [(0.25, 0.5), (0.5, 0.5)] = mitad en +25%, mitad en +50%.
    Lo que no esté en tiers sale por tiempo (o por stop).
    breakeven: después de la primera toma de ganancia, el stop sube al precio de entrada."""
    stop_pct: float = 0.25
    tiers: tuple[tuple[float, float], ...] = ((0.25, 0.5), (0.5, 0.5))
    breakeven: bool = True
    name: str = ""
    max_minutes: int | None = None  # salida por tiempo: si no funcionó en N minutos, fuera

    def label(self) -> str:
        if self.name:
            return self.name
        parts = [f"{int(f * 100)}% en +{int(g * 100)}%" for g, f in self.tiers]
        rest = 1 - sum(f for _, f in self.tiers)
        if rest > 1e-9:
            parts.append(f"{int(round(rest * 100))}% al cierre por tiempo")
        return (", ".join(parts) + (" | stop a breakeven" if self.breakeven else "")
                + (f" | máx {self.max_minutes} min" if self.max_minutes else ""))


EXIT_PLANS = [
    ExitPlan(tiers=((0.20, 1.0),), breakeven=False),
    ExitPlan(tiers=((0.30, 1.0),), breakeven=False),
    ExitPlan(tiers=((0.50, 1.0),), breakeven=False),
    ExitPlan(tiers=((0.25, 0.5), (0.50, 0.5)), breakeven=False),
    ExitPlan(tiers=((0.25, 0.5), (0.50, 0.5)), breakeven=True),
    ExitPlan(tiers=((0.20, 0.5), (0.40, 0.25), (0.80, 0.25)), breakeven=True),
    ExitPlan(tiers=((0.30, 0.5),), breakeven=True),  # mitad en +30%, el resto corre hasta la hora límite
    ExitPlan(tiers=((0.25, 0.34), (0.50, 0.33), (1.00, 0.33)), breakeven=True),
]

STOP_LEVELS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


@dataclass
class OptionPath:
    """Recorrido de la opción desde la entrada hasta la hora límite, sin stops."""
    entry: float
    strike: float
    mae: float           # peor caída (ej. -0.18 = llegó a perder 18%)
    mfe: float           # mejor ganancia
    mae_before: dict = field(default_factory=dict)  # {0.25: peor caída ANTES de tocar +25%} o None si nunca llegó


def _prices(ctx: DayContext, i: int, K: float, sigma: float, call: bool, dte: int) -> tuple[float, float, float]:
    """(precio en la apertura, peor precio y mejor precio de la opción) en la vela de 1 minuto `i`."""
    T = years_left(ctx.times[i], dte)
    f = bs_call if call else bs_put
    o = f(ctx.o[i], K, T, sigma)
    lo, hi = f(ctx.l[i], K, T, sigma), f(ctx.h[i], K, T, sigma)
    return o, (lo if call else hi), (hi if call else lo)


def _sell(price: float, slip: float) -> float:
    """Precio al vender: el valor teórico menos el costo por lado (medio spread + deslizamiento)."""
    return max(price - slip, 0.0)


def _entry(ctx: DayContext, sig: Signal, iv: float, delta: float, dte: int, slip: float):
    e = sig.pos + 1
    if e >= len(ctx.o):
        return None
    call = sig.direction == 1
    S = float(ctx.o[e])
    K = strike_for_delta(S, years_left(ctx.times[e], dte), iv, delta, call)
    mid = (bs_call if call else bs_put)(S, K, years_left(ctx.times[e], dte), iv)
    if mid < 0.05:
        return None
    return e, call, K, mid + slip


def option_path(ctx: DayContext, sig: Signal, iv: float = 0.18, delta: float = 0.45, dte: int = 0,
                slip: float = 0.02, gains=(0.20, 0.25, 0.30, 0.50)) -> OptionPath | None:
    start = _entry(ctx, sig, iv, delta, dte, slip)
    if start is None:
        return None
    e, call, K, cost = start
    worst_so_far, best = 0.0, 0.0
    mae_before = {g: None for g in gains}
    for i in range(e, len(ctx.o)):
        if i > e and ctx.times[i] >= sig.exit_by:
            break
        _, worst, hi = _prices(ctx, i, K, iv, call, dte)
        worst_so_far = min(worst_so_far, _sell(worst, slip) / cost - 1)
        gain = _sell(hi, slip) / cost - 1
        best = max(best, gain)
        for g in gains:
            if mae_before[g] is None and gain >= g:
                mae_before[g] = worst_so_far  # conservador: la caída de esta vela cuenta como anterior
    return OptionPath(entry=cost, strike=K, mae=worst_so_far, mfe=best, mae_before=mae_before)


def simulate_option(ctx: DayContext, sig: Signal, strategy: str, plan: ExitPlan, iv: float = 0.18,
                    delta: float = 0.45, dte: int = 0, slip: float = 0.02) -> Trade | None:
    """Simula la opción con stop sobre la prima y salidas escalonadas. Devuelve un Trade en R (1R = stop)."""
    start = _entry(ctx, sig, iv, delta, dte, slip)
    if start is None:
        return None
    e, call, K, cost = start
    stop = cost * (1 - plan.stop_pct)
    pending = list(plan.tiers)
    remaining = 1.0
    pnl = 0.0  # en fracción de la prima
    reasons = []
    exit_pos = len(ctx.o) - 1
    for i in range(e, len(ctx.o)):
        o, worst, best = _prices(ctx, i, K, iv, call, dte)
        if i > e and (ctx.times[i] >= sig.exit_by or (plan.max_minutes and i - e >= plan.max_minutes)):
            pnl += remaining * (_sell(o, slip) / cost - 1)
            reasons.append("tiempo")
            remaining, exit_pos = 0.0, i
            break
        if i > e and _sell(o, slip) <= stop:  # abrió por debajo del stop
            pnl += remaining * (_sell(o, slip) / cost - 1)
            reasons.append("stop" if stop < cost else "breakeven")
            remaining, exit_pos = 0.0, i
            break
        if _sell(worst, slip) <= stop:  # se revisa el stop antes que las ganancias (lo más conservador)
            pnl += remaining * (stop / cost - 1)
            reasons.append("stop" if stop < cost else "breakeven")
            remaining, exit_pos = 0.0, i
            break
        while pending and _sell(best, slip) >= cost * (1 + pending[0][0]):
            gain, frac = pending.pop(0)
            frac = min(frac, remaining)
            pnl += frac * gain
            remaining -= frac
            reasons.append(f"+{int(gain * 100)}%")
            if plan.breakeven:
                stop = max(stop, cost)
        if remaining <= 1e-9:
            exit_pos = i
            break
    else:
        last = len(ctx.o) - 1
        o_last = (bs_call if call else bs_put)(ctx.c[last], K, years_left(ctx.times[last], dte), iv)
        pnl += remaining * (_sell(o_last, slip) / cost - 1)
        reasons.append("cierre")

    r = pnl / plan.stop_pct
    return Trade(day=ctx.day, strategy=strategy, direction=sig.direction, entry_time=ctx.times[e],
                 entry=round(cost, 3), stop=round(cost * (1 - plan.stop_pct), 3), target=float(K),
                 exit_time=ctx.times[exit_pos], exit=round(cost * (1 + pnl), 3),
                 exit_reason=" / ".join(reasons), r=round(r, 3), r_net=round(r, 3),
                 gap_pct=round(ctx.gap_pct, 3), note=sig.note, conditions=sig.conditions)


def simulate_option_levels(ctx: DayContext, sig: Signal, strategy: str, iv: float = 0.18, delta: float = 0.45,
                           dte: int = 0, slip: float = 0.02, max_minutes: int | None = None) -> Trade | None:
    """La opción sale cuando el ETF toca el stop o el objetivo de la estrategia (o por tiempo),
    igual que en el backtest normal. 1R = lo que pierde la opción si el ETF toca el stop."""
    start = _entry(ctx, sig, iv, delta, dte, slip)
    if start is None:
        return None
    e, call, K, cost = start
    f = bs_call if call else bs_put
    d = sig.direction
    S0 = float(ctx.o[e])
    risk_u = (S0 - sig.stop) * d
    if risk_u <= 0:
        return None
    target_u = sig.target_price if sig.target_price is not None else S0 + d * (sig.target_r or 1.5) * risk_u
    if (target_u - S0) * d <= 0:
        return None
    risk = cost - _sell(f(sig.stop, K, years_left(ctx.times[e], dte), iv), slip)  # pérdida si toca el stop ya
    if risk <= 0:
        return None
    exit_price, exit_pos, reason = None, len(ctx.o) - 1, "cierre"
    for i in range(e, len(ctx.o)):
        T = years_left(ctx.times[i], dte)
        if i > e and (ctx.times[i] >= sig.exit_by or (max_minutes and i - e >= max_minutes)):
            exit_price, exit_pos, reason = f(ctx.o[i], K, T, iv), i, "tiempo"
            break
        worst, best = (ctx.l[i], ctx.h[i]) if d == 1 else (ctx.h[i], ctx.l[i])
        if (worst - sig.stop) * d <= 0:
            level = ctx.o[i] if i > e and (ctx.o[i] - sig.stop) * d <= 0 else sig.stop
            exit_price, exit_pos, reason = f(level, K, T, iv), i, "stop"
            break
        if (best - target_u) * d >= 0:
            exit_price, exit_pos, reason = f(target_u, K, T, iv), i, "objetivo"
            break
    if exit_price is None:
        last = len(ctx.o) - 1
        exit_price = f(ctx.c[last], K, years_left(ctx.times[last], dte), iv)
    exit_price = _sell(exit_price, slip)
    r = (exit_price - cost) / risk
    return Trade(day=ctx.day, strategy=strategy, direction=d, entry_time=ctx.times[e], entry=round(cost, 3),
                 stop=round(cost - risk, 3), target=float(K), exit_time=ctx.times[exit_pos],
                 exit=round(exit_price, 3), exit_reason=reason, r=round(r, 3), r_net=round(r, 3),
                 gap_pct=round(ctx.gap_pct, 3), note=sig.note, conditions=sig.conditions)


def strike_step(price: float, etf: bool = True) -> float:
    """Separación típica entre strikes: $1 en ETFs; en acciones depende del precio."""
    if etf or price < 200:
        return 1.0
    return 2.5 if price < 400 else 5.0


def simulate_spread_levels(ctx: DayContext, sig: Signal, strategy: str, iv: float = 0.18, delta: float = 0.70,
                           dte: int = 2, slip: float = 0.02, short: str = "target",
                           step: float | None = None, etf: bool = True) -> Trade | None:
    """Spread de débito: compra la opción ITM (delta `delta`) y vende otra más afuera.
    short='target': vende el strike del objetivo de la estrategia | short='atm': el strike más cercano al precio.
    Sale cuando el activo toca el stop o el objetivo (o por tiempo). Se paga `slip` por pata y por lado.
    1R = lo que pierde el spread si el activo toca el stop."""
    e = sig.pos + 1
    if e >= len(ctx.o):
        return None
    d = sig.direction
    call = d == 1
    f = bs_call if call else bs_put
    S0 = float(ctx.o[e])
    step = step or strike_step(S0, etf)
    T0 = years_left(ctx.times[e], dte)
    K1 = strike_for_delta(S0, T0, iv, delta, call, step)
    risk_u = (S0 - sig.stop) * d
    if risk_u <= 0:
        return None
    target_u = sig.target_price if sig.target_price is not None else S0 + d * (sig.target_r or 1.5) * risk_u
    if (target_u - S0) * d <= 0:
        return None
    if short == "atm":
        K2 = round(S0 / step) * step
    else:
        K2 = math.ceil(target_u / step) * step if call else math.floor(target_u / step) * step
    if (K2 - K1) * d < step:
        K2 = K1 + d * step  # al menos un strike de ancho

    def value(S: float, T: float) -> float:
        return f(S, K1, T, iv) - f(S, K2, T, iv)

    cost = value(S0, T0) + 2 * slip
    width = abs(K2 - K1)
    if cost >= width or cost <= 0.05:
        return None  # el spread no tiene sentido (cuesta más que su valor máximo)
    risk = cost - max(value(sig.stop, T0) - 2 * slip, 0.0)
    if risk <= 0:
        return None
    exit_value, exit_pos, reason = None, len(ctx.o) - 1, "cierre"
    for i in range(e, len(ctx.o)):
        T = years_left(ctx.times[i], dte)
        if i > e and ctx.times[i] >= sig.exit_by:
            exit_value, exit_pos, reason = value(ctx.o[i], T), i, "tiempo"
            break
        worst, best = (ctx.l[i], ctx.h[i]) if call else (ctx.h[i], ctx.l[i])
        if (worst - sig.stop) * d <= 0:
            level = ctx.o[i] if i > e and (ctx.o[i] - sig.stop) * d <= 0 else sig.stop
            exit_value, exit_pos, reason = value(level, T), i, "stop"
            break
        if (best - target_u) * d >= 0:
            exit_value, exit_pos, reason = value(target_u, T), i, "objetivo"
            break
    if exit_value is None:
        last = len(ctx.o) - 1
        exit_value = value(ctx.c[last], years_left(ctx.times[last], dte))
    exit_value = max(exit_value - 2 * slip, 0.0)
    r = (exit_value - cost) / risk
    return Trade(day=ctx.day, strategy=strategy, direction=d, entry_time=ctx.times[e], entry=round(cost, 3),
                 stop=round(cost - risk, 3), target=float(K2), exit_time=ctx.times[exit_pos],
                 exit=round(exit_value, 3), exit_reason=reason, r=round(r, 3), r_net=round(r, 3),
                 gap_pct=round(ctx.gap_pct, 3), note=f"spread {K1:g}/{K2:g} | {sig.note}", conditions=sig.conditions)


def signals_for(contexts: list[DayContext], strategy_fn, params: dict) -> list[tuple[DayContext, Signal]]:
    out = []
    for ctx in contexts:
        sig = strategy_fn(ctx, **params)
        if sig is not None:
            out.append((ctx, sig))
    return out


def stop_study(pairs, iv: float = 0.18, slip: float = 0.02, delta: float = 0.45, dte: int = 0) -> dict:
    """¿Qué tan profundo cae la opción antes de ganar? Sirve para elegir el stop."""
    paths = [p for p in (option_path(ctx, sig, iv=iv, delta=delta, dte=dte, slip=slip) for ctx, sig in pairs) if p]
    rows = []
    for g in (0.20, 0.25, 0.30, 0.50):
        reached = [p.mae_before[g] for p in paths if p.mae_before[g] is not None]
        row = {"meta": f"+{int(g * 100)}%", "trades_que_llegan": len(reached),
               "%_del_total": round(len(reached) / len(paths), 2) if paths else 0}
        for s in STOP_LEVELS:
            # de los que llegaron a la meta, cuántos habrían sido sacados antes por este stop
            row[f"sacados_stop_{int(s * 100)}%"] = (round(float(np.mean([m <= -s for m in reached])), 2)
                                                    if reached else None)
        rows.append(row)
    maes = [p.mae for p in paths]
    return {
        "paths": len(paths),
        "mae_percentiles": {f"p{q}": round(float(np.percentile(maes, q)), 3) for q in (25, 50, 75, 90)} if maes else {},
        "table": rows,
        "avg_premium": round(float(np.mean([p.entry for p in paths])), 2) if paths else 0,
    }
