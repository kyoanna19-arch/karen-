"""Motor de backtest: simula las señales, mide resultados en R y los desglosa por mes.

R = múltiplo del riesgo. Si arriesgas $80 y ganas $160, ese trade fue +2R.
Medimos en R sobre el subyacente porque los precios históricos de opciones son caros de
conseguir; una opción ATM se mueve aproximadamente con el subyacente (delta ~0.5), pero
el spread bid/ask y el theta cuestan. Eso se descuenta con `cost_r` en cada trade.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from datetime import date, time
from pathlib import Path

import pandas as pd

from .strategies import STRATEGIES, DayContext, Signal

REGULAR_START, REGULAR_END = "09:30", "15:59"
PREMARKET_START, PREMARKET_END = "04:00", "09:29"


# --------------------------------------------------------------------------- datos

def load_bars_csv(path: str | Path) -> pd.DataFrame:
    """CSV con columnas: datetime, open, high, low, close, volume (hora del Este)."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    time_col = next((cols[c] for c in ("datetime", "time", "timestamp", "date") if c in cols), None)
    if time_col is None:
        raise ValueError("El CSV necesita una columna 'datetime'")
    df.index = pd.to_datetime(df.pop(time_col))
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    df.columns = [c.lower().strip() for c in df.columns]
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")
    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


def build_contexts(bars: pd.DataFrame, min_regular_bars: int = 120) -> list[DayContext]:
    """Divide barras de 1 minuto en días. El primer día solo sirve de referencia (cierre previo)."""
    contexts: list[DayContext] = []
    prev = None
    for day, day_bars in bars.groupby(bars.index.date):
        regular = day_bars.between_time(REGULAR_START, REGULAR_END)
        if len(regular) < min_regular_bars:
            continue
        if prev is not None:
            contexts.append(DayContext(
                day=day, bars=regular, premarket=day_bars.between_time(PREMARKET_START, PREMARKET_END),
                prev_close=prev["close"], prev_high=prev["high"], prev_low=prev["low"]))
        prev = {"close": float(regular["close"].iloc[-1]), "high": float(regular["high"].max()),
                "low": float(regular["low"].min())}
    return contexts


# --------------------------------------------------------------------------- simulación

@dataclass
class Trade:
    day: date
    strategy: str
    direction: int
    entry_time: time
    entry: float
    stop: float
    target: float
    exit_time: time
    exit: float
    exit_reason: str
    r: float          # resultado bruto en R
    r_net: float      # resultado después de costos
    gap_pct: float
    note: str


def simulate(ctx: DayContext, sig: Signal, strategy: str, cost_r: float = 0.1) -> Trade | None:
    """Entra en la apertura de la barra siguiente a la señal y sale por stop, objetivo o tiempo.
    Si en la misma barra se tocan stop y objetivo, asumimos lo peor (stop)."""
    e = sig.pos + 1
    if e >= len(ctx.o):
        return None
    d = sig.direction
    entry = float(ctx.o[e])
    risk = (entry - sig.stop) * d
    if risk <= 0:
        return None  # el precio ya pasó el stop antes de poder entrar
    target = sig.target_price if sig.target_price is not None else entry + d * sig.target_r * risk
    if (target - entry) * d <= 0:
        return None  # el objetivo ya se alcanzó

    exit_price, exit_pos, reason = float(ctx.c[-1]), len(ctx.c) - 1, "cierre"
    for i in range(e, len(ctx.o)):
        o, h, l = ctx.o[i], ctx.h[i], ctx.l[i]
        if i > e and ctx.times[i] >= sig.exit_by:
            exit_price, exit_pos, reason = float(o), i, "tiempo"
            break
        worst, best = (l, h) if d == 1 else (h, l)
        if i > e and (o - sig.stop) * d <= 0:  # abrió más allá del stop
            exit_price, exit_pos, reason = float(o), i, "stop"
            break
        if (worst - sig.stop) * d <= 0:
            exit_price, exit_pos, reason = sig.stop, i, "stop"
            break
        if (best - target) * d >= 0:
            exit_price, exit_pos, reason = float(target), i, "objetivo"
            break

    r = (exit_price - entry) * d / risk
    return Trade(day=ctx.day, strategy=strategy, direction=d, entry_time=ctx.times[e], entry=entry,
                 stop=sig.stop, target=float(target), exit_time=ctx.times[exit_pos], exit=exit_price,
                 exit_reason=reason, r=round(r, 3), r_net=round(r - cost_r, 3),
                 gap_pct=round(ctx.gap_pct, 3), note=sig.note)


def run_strategy(contexts: list[DayContext], name: str, params: dict | None = None,
                 cost_r: float = 0.1) -> list[Trade]:
    spec = STRATEGIES[name]
    trades = []
    for ctx in contexts:
        sig = spec.fn(ctx, **(params or {}))
        if sig is not None:
            trade = simulate(ctx, sig, name, cost_r)
            if trade is not None:
                trades.append(trade)
    return trades


# --------------------------------------------------------------------------- estadísticas

def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])


def monthly_table(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["trades", "win_rate", "total_r"])
    df = trades_frame(trades)
    df["month"] = pd.to_datetime(df["day"]).dt.strftime("%Y-%m")
    g = df.groupby("month")["r_net"]
    return pd.DataFrame({"trades": g.size(), "win_rate": g.apply(lambda s: (s > 0).mean()).round(2),
                         "total_r": g.sum().round(2)})


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0, "profit_factor": 0.0,
                "max_drawdown_r": 0.0, "months": 0, "pct_months_positive": 0.0, "worst_month_r": 0.0,
                "score": 0.0}
    r = pd.Series([t.r_net for t in trades])
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    equity = r.cumsum()
    drawdown = (equity.cummax().clip(lower=0) - equity).max()
    months = monthly_table(trades)
    avg = r.mean()
    return {
        "trades": len(r),
        "win_rate": round((r > 0).mean(), 3),
        "avg_r": round(avg, 3),
        "total_r": round(r.sum(), 2),
        "profit_factor": round(wins / losses, 2) if losses > 0 else float("inf"),
        "max_drawdown_r": round(float(drawdown), 2),
        "months": len(months),
        "pct_months_positive": round((months["total_r"] > 0).mean(), 2),
        "worst_month_r": round(float(months["total_r"].min()), 2),
        # calidad = expectativa × √trades: premia una ventaja consistente, no un par de trades con suerte
        "score": round(avg * math.sqrt(len(r)), 3),
    }


def param_combinations(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def optimize(contexts: list[DayContext], names: list[str] | None = None, cost_r: float = 0.1,
             min_trades: int = 10) -> pd.DataFrame:
    """Prueba cada combinación de parámetros de cada estrategia y las ordena por calidad."""
    rows = []
    for name in names or list(STRATEGIES):
        spec = STRATEGIES[name]
        for params in param_combinations(spec.grid):
            if spec.valid is not None and not spec.valid(params):
                continue
            stats = summarize(run_strategy(contexts, name, params, cost_r))
            if stats["trades"] >= min_trades:
                rows.append({"strategy": name, "params": params, **stats})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def walk_forward(contexts: list[DayContext], train_frac: float = 0.7, cost_r: float = 0.1,
                 min_trades: int = 10, names: list[str] | None = None) -> pd.DataFrame:
    """Elige los mejores parámetros con los primeros meses (entrenamiento) y los prueba en meses que
    la optimización nunca vio (prueba). Si una estrategia solo funciona en entrenamiento, es suerte."""
    split = int(len(contexts) * train_frac)
    train, test = contexts[:split], contexts[split:]
    rows = []
    for name in names or list(STRATEGIES):
        ranked = optimize(train, [name], cost_r, min_trades)
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        test_stats = summarize(run_strategy(test, name, best["params"], cost_r))
        rows.append({
            "strategy": name, "params": best["params"],
            "train_trades": best["trades"], "train_avg_r": best["avg_r"], "train_total_r": best["total_r"],
            "test_trades": test_stats["trades"], "test_avg_r": test_stats["avg_r"],
            "test_total_r": test_stats["total_r"], "test_win_rate": test_stats["win_rate"],
            "test_pct_months_positive": test_stats["pct_months_positive"],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("test_avg_r", ascending=False).reset_index(drop=True)


def breakdown(trades: list[Trade]) -> dict[str, pd.DataFrame]:
    """¿En qué condiciones funciona? Resultados por tamaño de gap, día de la semana y dirección."""
    if not trades:
        return {}
    df = trades_frame(trades)
    df["gap"] = pd.cut(df["gap_pct"], [-100, -1, -0.3, 0.3, 1, 100],
                       labels=["< -1%", "-1% a -0.3%", "plano", "0.3% a 1%", "> 1%"])
    df["weekday"] = pd.to_datetime(df["day"]).dt.day_name()
    df["side"] = df["direction"].map({1: "CALL", -1: "PUT"})
    df["hour"] = df["entry_time"].map(lambda t: f"{t.hour:02d}:{0 if t.minute < 30 else 30:02d}")

    def agg(col: str) -> pd.DataFrame:
        g = df.groupby(col, observed=True)["r_net"]
        return pd.DataFrame({"trades": g.size(), "win_rate": g.apply(lambda s: (s > 0).mean()).round(2),
                             "avg_r": g.mean().round(3), "total_r": g.sum().round(2)})

    return {"Por gap": agg("gap"), "Por día": agg("weekday"), "Por dirección": agg("side"),
            "Por hora de entrada": agg("hour")}


def ablation(contexts: list[DayContext], name: str, params: dict | None = None,
             cost_r: float = 0.1) -> pd.DataFrame:
    """Compara la estrategia completa contra versiones con una regla apagada.
    Si quitar una regla NO empeora (o mejora) el resultado, esa regla no está aportando."""
    base = dict(params or {})
    variants = {"completa": {}, **STRATEGIES[name].ablations}
    rows = []
    for label, override in variants.items():
        stats = summarize(run_strategy(contexts, name, {**base, **override}, cost_r))
        rows.append({"variante": label, **{k: stats[k] for k in (
            "trades", "win_rate", "avg_r", "total_r", "profit_factor", "pct_months_positive", "worst_month_r")}})
    return pd.DataFrame(rows)
