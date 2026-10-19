"""Prueba UNA configuración fija de la estrategia en varios activos y resume todo en una tabla.

Cada activo es una prueba independiente: si la estrategia gana en varios, la ventaja es más
creíble; si solo gana en uno, probablemente fue suerte.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import backtest as bt
from . import options_sim as osim
from .strategies import STRATEGIES

# Volatilidad implícita típica (supuesto aproximado) y costo por lado en dólares (medio spread + deslizamiento)
TYPICAL_IV = {"SPY": 0.18, "QQQ": 0.20, "IWM": 0.24, "DIA": 0.16, "AAPL": 0.27, "MSFT": 0.26, "NVDA": 0.48,
              "AMD": 0.50, "META": 0.38, "TSLA": 0.58, "AMZN": 0.33, "GOOGL": 0.32}
ETFS = {"SPY", "QQQ", "IWM", "DIA"}


def default_iv(symbol: str) -> float:
    return TYPICAL_IV.get(symbol.upper(), 0.35)


def default_slip(symbol: str) -> float:
    return 0.02 if symbol.upper() in ETFS else 0.05


def _year_avg(trades, year: int) -> float | None:
    rs = [t.r_net for t in trades if t.day.year == year]
    return round(sum(rs) / len(rs), 3) if rs else None


STRUCTURES = ("CALL", "spread objetivo", "spread ATM")


def _simulate(structure: str, c, sig, strategy: str, iv: float, delta: float, dte: int, slip: float, etf: bool):
    if structure == "CALL":
        return osim.simulate_option_levels(c, sig, strategy, iv=iv, delta=delta, dte=dte, slip=slip)
    short = "target" if structure == "spread objetivo" else "atm"
    return osim.simulate_spread_levels(c, sig, strategy, iv=iv, delta=delta, dte=dte, slip=slip, short=short, etf=etf)


def evaluate_symbol(symbol: str, contexts, strategy: str, params: dict, dte: int, delta: float,
                    iv: float, slip: float, risk: float, account: float, max_position_pct: float,
                    structures=STRUCTURES) -> tuple[list[dict], dict[str, list]]:
    etf_stats = bt.summarize(bt.run_strategy(contexts, strategy, params, cost_r=0.1))
    pairs = osim.signals_for(contexts, STRATEGIES[strategy].fn, params)
    budget = account * max_position_pct / 100
    rows, trades_by_structure = [], {}
    for structure in structures:
        trades = [t for t in (_simulate(structure, c, s, strategy, iv, delta, dte, slip, symbol in ETFS)
                              for c, s in pairs) if t]
        st = bt.summarize(trades)
        premium = sum(t.entry for t in trades) / len(trades) if trades else 0.0
        row = {
            "activo": symbol, "estructura": structure, "trades": st["trades"],
            "ETF_avg_r": etf_stats["avg_r"] if structure == "CALL" else "",
            "avg_r": st["avg_r"], f"$_por_trade_{risk:.0f}": round(st["avg_r"] * risk, 2),
            "PF": st["profit_factor"], "meses_+": st["pct_months_positive"], "peor_racha_r": st["max_drawdown_r"],
            "costo_contrato": round(premium * 100), "cabe": "sí" if premium * 100 <= budget else "NO",
        }
        for y in sorted({t.day.year for t in trades})[-2:]:
            row[f"avg_r_{y}"] = _year_avg(trades, y)
        rows.append(row)
        trades_by_structure[structure] = trades
    return rows, trades_by_structure


def compare(symbols: list[str], data_dir: str | Path = "data", strategy: str = "karen_pullback",
            params: dict | None = None, dte: int = 2, delta: float = 0.70, risk: float = 150,
            account: float = 3000, max_position_pct: float = 30, iv_override: float | None = None,
            slip_override: float | None = None) -> tuple[pd.DataFrame, list[str]]:
    rows, missing = [], []
    all_trades: dict[str, list] = {k: [] for k in STRUCTURES}
    for symbol in symbols:
        path = Path(data_dir) / f"{symbol.upper()}_1min.csv"
        if not path.exists():
            missing.append(symbol)
            continue
        contexts = bt.build_contexts(bt.load_bars_csv(path))
        sym_rows, trades = evaluate_symbol(symbol.upper(), contexts, strategy, params or {}, dte, delta,
                                           iv_override or default_iv(symbol), slip_override or default_slip(symbol),
                                           risk, account, max_position_pct)
        rows += sym_rows
        for k, v in trades.items():
            all_trades[k] += v
    for structure, trades in all_trades.items():
        if trades:
            total = bt.summarize(trades)
            rows.append({"activo": "TODOS", "estructura": structure, "trades": total["trades"],
                         "avg_r": total["avg_r"], f"$_por_trade_{risk:.0f}": round(total["avg_r"] * risk, 2),
                         "PF": total["profit_factor"], "meses_+": total["pct_months_positive"],
                         "peor_racha_r": total["max_drawdown_r"]})
    return pd.DataFrame(rows).fillna(""), missing
