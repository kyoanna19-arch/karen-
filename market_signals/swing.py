"""Swing trading con velas diarias: trades de varios días en vez de minutos.

Con movimientos de varios días, el spread y las comisiones pesan mucho menos que en scalping.
Además, al quedarse de un día para otro, NO cuenta como day trade (no aplica la regla PDT).

Reglas probadas (todas compran solo en tendencia alcista de largo plazo: cierre > promedio de 200 días):
  rsi2   : el RSI de 2 días cae por debajo de `entry_level` (caída fuerte de corto plazo)
  down3  : 3 cierres seguidos más bajos
  ibs    : cierra en la parte baja del rango del día (IBS < `entry_level` / 100)
Salida: el cierre supera el promedio de 5 días, o pasan `max_days` días.
La señal se calcula con el cierre del día; la compra/venta se hace en la APERTURA siguiente.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .strategies import rsi

ETFS = {"SPY", "QQQ", "IWM", "DIA"}


# --------------------------------------------------------------------------- datos

def daily_path(symbol: str, folder: str | Path = "data") -> Path:
    return Path(folder) / f"{symbol.upper()}_1d.csv"


def download_daily(symbol: str, start: str = "2005-01-01", folder: str | Path = "data") -> Path:
    """Descarga velas diarias ajustadas por dividendos y splits desde Yahoo Finance (gratis)."""
    import yfinance as yf
    df = yf.Ticker(symbol.upper()).history(start=start, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"Yahoo no devolvió datos para {symbol}")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    out = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    path = daily_path(symbol, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")
    return path


def load_daily(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


# --------------------------------------------------------------------------- reglas

@dataclass(frozen=True)
class SwingRule:
    rule: str = "rsi2"
    entry_level: float = 10
    exit_rule: str = "sma5"      # "sma5" o "rsi70"
    max_days: int = 10
    trend_filter: bool = True    # solo comprar si el cierre está sobre el promedio de 200 días

    def label(self) -> str:
        return (f"{self.rule} {self.entry_level:g} | salida {self.exit_rule} | máx {self.max_days}d"
                + ("" if self.trend_filter else " | sin filtro 200d"))


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma200"] = out["close"].rolling(200).mean()
    out["sma5"] = out["close"].rolling(5).mean()
    out["rsi2"] = rsi(out["close"].to_numpy(), 2)
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["ibs"] = ((out["close"] - out["low"]) / rng).fillna(0.5)
    out["down"] = (out["close"] < out["close"].shift(1)).astype(int)
    out["down3"] = out["down"].rolling(3).sum() == 3
    return out


def entry_signal(row: pd.Series, rule: SwingRule) -> bool:
    if rule.trend_filter and not (row["close"] > row["sma200"]):
        return False
    if rule.rule == "rsi2":
        return row["rsi2"] < rule.entry_level
    if rule.rule == "down3":
        return bool(row["down3"])
    if rule.rule == "ibs":
        return row["ibs"] < rule.entry_level / 100
    raise ValueError(f"Regla desconocida: {rule.rule}")


def exit_signal(row: pd.Series, rule: SwingRule) -> bool:
    if rule.exit_rule == "sma5":
        return row["close"] > row["sma5"]
    if rule.exit_rule == "rsi70":
        return row["rsi2"] > 70
    raise ValueError(f"Salida desconocida: {rule.exit_rule}")


# --------------------------------------------------------------------------- backtest

@dataclass
class SwingTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry: float
    exit: float
    days: int
    ret: float  # rendimiento neto del trade (0.012 = +1.2%)


def backtest(df: pd.DataFrame, rule: SwingRule, cost: float = 0.0002) -> tuple[list[SwingTrade], pd.Series]:
    """Una posición a la vez, con toda la cuenta. `cost` = costo por lado (0.0002 = 0.02%).
    Devuelve los trades y la curva de capital diaria (empieza en 1.0)."""
    d = indicators(df).dropna(subset=["sma200"] if rule.trend_filter else ["sma5"])
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    dates = d.index
    rows = [d.iloc[i] for i in range(len(d))]
    trades: list[SwingTrade] = []
    equity = np.ones(len(d))
    cash, shares, entry_i, entry_price = 1.0, 0.0, None, 0.0
    pending_entry = pending_exit = False
    for i in range(len(d)):
        if pending_exit and shares:
            cash = shares * o[i] * (1 - cost)
            ret = o[i] * (1 - cost) / (entry_price * (1 + cost)) - 1
            trades.append(SwingTrade(dates[entry_i], dates[i], entry_price, o[i], i - entry_i, ret))
            shares, pending_exit = 0.0, False
        elif pending_entry and not shares:
            entry_price = o[i]
            shares = cash / (o[i] * (1 + cost))
            cash, entry_i, pending_entry = 0.0, i, False
        equity[i] = cash + shares * c[i]
        if shares:
            if exit_signal(rows[i], rule) or i - entry_i >= rule.max_days:
                pending_exit = True
        elif entry_signal(rows[i], rule):
            pending_entry = True
    return trades, pd.Series(equity, index=dates)


def stats(trades: list[SwingTrade], equity: pd.Series, df: pd.DataFrame) -> dict:
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    dd = (equity / equity.cummax() - 1).min()
    rets = np.array([t.ret for t in trades])
    bh = df.loc[equity.index[0]:, "close"]
    bh_curve = bh / bh.iloc[0]
    return {
        "trades": len(trades),
        "trades_por_año": round(len(trades) / years, 1),
        "ganadores": round(float((rets > 0).mean()), 2) if len(rets) else 0.0,
        "prom_%": round(float(rets.mean()) * 100, 2) if len(rets) else 0.0,
        "días_prom": round(float(np.mean([t.days for t in trades])), 1) if trades else 0.0,
        "CAGR_%": round((equity.iloc[-1] ** (1 / years) - 1) * 100, 1),
        "caída_máx_%": round(float(dd) * 100, 1),
        "tiempo_invertido_%": round(sum(t.days for t in trades) / len(equity) * 100),
        "comprar_y_mantener_CAGR_%": round((bh_curve.iloc[-1] ** (1 / years) - 1) * 100, 1),
        "comprar_y_mantener_caída_%": round(float((bh_curve / bh_curve.cummax() - 1).min()) * 100, 1),
    }


def yearly(trades: list[SwingTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame({"año": [t.exit_date.year for t in trades], "ret": [t.ret for t in trades]})
    g = df.groupby("año")["ret"]
    return pd.DataFrame({"trades": g.size(), "ganadores": g.apply(lambda s: (s > 0).mean()).round(2),
                         "resultado_%": g.apply(lambda s: (np.prod(1 + s) - 1) * 100).round(1)})


DEFAULT_RULE = SwingRule()  # rsi2 < 10, salida sobre el promedio de 5 días, máx 10 días, con filtro 200d

GRID = [SwingRule(r, lvl, ex, md) for r, lvl, ex, md in itertools.product(
    ["rsi2"], [5, 10, 15], ["sma5", "rsi70"], [5, 10])] + [
    SwingRule("down3", 0, "sma5", 10), SwingRule("ibs", 20, "sma5", 10), SwingRule("rsi2", 10, "sma5", 10, False)]


def cost_for(symbol: str) -> float:
    return 0.0002 if symbol.upper() in ETFS else 0.0005


def compare_symbols(symbols: list[str], rule: SwingRule = DEFAULT_RULE, folder: str | Path = "data"):
    rows, missing = [], []
    for sym in symbols:
        path = daily_path(sym, folder)
        if not path.exists():
            missing.append(sym)
            continue
        df = load_daily(path)
        trades, equity = backtest(df, rule, cost_for(sym))
        st = stats(trades, equity, df)
        half = equity.index[0] + (equity.index[-1] - equity.index[0]) / 2
        first = [t.ret for t in trades if t.exit_date < half]
        second = [t.ret for t in trades if t.exit_date >= half]
        rows.append({"activo": sym.upper(), "desde": equity.index[0].year, **st,
                     "prom_%_1a_mitad": round(float(np.mean(first)) * 100, 2) if first else None,
                     "prom_%_2a_mitad": round(float(np.mean(second)) * 100, 2) if second else None})
    return pd.DataFrame(rows), missing


def grid_table(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rows = []
    for rule in GRID:
        trades, equity = backtest(df, rule, cost_for(symbol))
        st = stats(trades, equity, df)
        rows.append({"regla": rule.label(), **{k: st[k] for k in ("trades", "ganadores", "prom_%", "días_prom",
                                                                  "CAGR_%", "caída_máx_%", "tiempo_invertido_%")}})
    return pd.DataFrame(rows)


def shares_for(account: float, price: float) -> int:
    return math.floor(account / price)
