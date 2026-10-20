"""¿Tiene poder de predicción la señal, SIN stop y sin reglas de salida?

Mide cuánto se movió el precio después de cada señal en varios plazos (minutos, cierre del día,
días siguientes) y lo compara contra entrar en momentos al azar en el mismo horario.
Si la señal no supera al azar en ningún plazo, ningún cambio de stop o de salida la va a arreglar.
"""
from __future__ import annotations

import math
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from .strategies import STRATEGIES, DayContext

MINUTE_HORIZONS = (5, 15, 30, 60)
LABELS = ["5min", "15min", "30min", "60min", "12:15", "cierre", "+1día", "+2días", "+5días"]


def _forward(contexts: list[DayContext], j: int, e: int, direction: int) -> dict[str, float]:
    """Rendimiento (%) desde la apertura de la vela e del día j hasta cada plazo."""
    ctx = contexts[j]
    entry = float(ctx.o[e])
    out = {}
    for h in MINUTE_HORIZONS:
        k = e + h - 1
        if k < len(ctx.c):
            out[f"{h}min"] = (ctx.c[k] / entry - 1) * 100 * direction
    noon = ctx.first_pos_at_or_after(time(12, 15)) - 1
    if noon > e:
        out["12:15"] = (ctx.c[noon] / entry - 1) * 100 * direction
    out["cierre"] = (ctx.c[-1] / entry - 1) * 100 * direction
    for days, label in ((1, "+1día"), (2, "+2días"), (5, "+5días")):
        if j + days < len(contexts):
            out[label] = (contexts[j + days].c[-1] / entry - 1) * 100 * direction
    return out


def signal_returns(contexts: list[DayContext], strategy: str, params: dict) -> pd.DataFrame:
    rows = []
    fn = STRATEGIES[strategy].fn
    for j, ctx in enumerate(contexts):
        sig = fn(ctx, **params)
        if sig is None or sig.pos + 1 >= len(ctx.o):
            continue
        rows.append({"day": ctx.day, **_forward(contexts, j, sig.pos + 1, sig.direction)})
    return pd.DataFrame(rows)


def baseline_returns(contexts: list[DayContext], start: str = "09:35", end: str = "11:55",
                     every: int = 5) -> pd.DataFrame:
    """Entradas 'al azar': cada `every` minutos en el mismo horario en que da señal la estrategia (CALL)."""
    rows = []
    t0, t1 = time.fromisoformat(start), time.fromisoformat(end)
    for j, ctx in enumerate(contexts):
        for e in range(ctx.first_pos_at_or_after(t0), len(ctx.times), every):
            if ctx.times[e] > t1:
                break
            rows.append(_forward(contexts, j, e, 1))
    return pd.DataFrame(rows)


def summarize_returns(df: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for label in LABELS:
        if label not in df:
            continue
        r = df[label].dropna().to_numpy()
        if len(r) < 2:
            continue
        std = r.std(ddof=1)
        out[label] = {"n": len(r), "prom_%": r.mean(), "ganadores": (r > 0).mean(),
                      "t": r.mean() / std * math.sqrt(len(r)) if std > 0 else 0.0}
    return out


def study(symbols: list[str], strategy: str, params: dict, data_dir: str | Path = "data"):
    """Devuelve (tabla por activo, tabla combinada, faltantes)."""
    per_symbol, pooled_sig, pooled_base, missing = [], [], [], []
    for sym in symbols:
        path = Path(data_dir) / f"{sym.upper()}_1min.csv"
        if not path.exists():
            missing.append(sym)
            continue
        contexts = bt.build_contexts(bt.load_bars_csv(path))
        sig = signal_returns(contexts, strategy, params)
        base = baseline_returns(contexts)
        if sig.empty:
            continue
        s, b = summarize_returns(sig), summarize_returns(base)
        row = {"activo": sym.upper(), "señales": len(sig)}
        for label in LABELS:
            if label in s and label in b:
                row[label] = round(s[label]["prom_%"] - b[label]["prom_%"], 3)
        per_symbol.append(row)
        pooled_sig.append(sig.assign(activo=sym.upper()))
        pooled_base.append(base)
    if not pooled_sig:
        return pd.DataFrame(), pd.DataFrame(), missing
    s = summarize_returns(pd.concat(pooled_sig))
    b = summarize_returns(pd.concat(pooled_base))
    pooled = []
    for label in LABELS:
        if label in s and label in b:
            edge = s[label]["prom_%"] - b[label]["prom_%"]
            positive_symbols = sum(1 for r in per_symbol if r.get(label, 0) > 0)
            pooled.append({"plazo": label, "señales": s[label]["n"],
                           "señal_prom_%": round(s[label]["prom_%"], 3),
                           "azar_prom_%": round(b[label]["prom_%"], 3),
                           "ventaja_%": round(edge, 3),
                           "señal_ganadores": round(s[label]["ganadores"], 2),
                           "azar_ganadores": round(b[label]["ganadores"], 2),
                           "t_señal": round(s[label]["t"], 1),
                           f"activos_con_ventaja_de_{len(per_symbol)}": positive_symbols})
    return pd.DataFrame(per_symbol), pd.DataFrame(pooled), missing
