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


def evaluate_symbol(symbol: str, contexts, strategy: str, params: dict, dte: int, delta: float,
                    iv: float, slip: float, risk: float, account: float, max_position_pct: float) -> dict:
    etf_trades = bt.run_strategy(contexts, strategy, params, cost_r=0.1)
    etf = bt.summarize(etf_trades)
    pairs = osim.signals_for(contexts, STRATEGIES[strategy].fn, params)
    opt_trades = [t for t in (osim.simulate_option_levels(c, s, strategy, iv=iv, delta=delta, dte=dte, slip=slip)
                              for c, s in pairs) if t]
    opt = bt.summarize(opt_trades)
    premium = sum(t.entry for t in opt_trades) / len(opt_trades) if opt_trades else 0.0
    budget = account * max_position_pct / 100
    years = sorted({t.day.year for t in opt_trades})
    row = {
        "activo": symbol, "trades": opt["trades"],
        "ETF_avg_r": etf["avg_r"], "ETF_PF": etf["profit_factor"],
        "opc_avg_r": opt["avg_r"], f"$_por_trade_{risk:.0f}": round(opt["avg_r"] * risk, 2),
        "opc_PF": opt["profit_factor"], "meses_+": opt["pct_months_positive"],
        "peor_racha_r": opt["max_drawdown_r"],
        "prima_prom": round(premium, 2), "cabe_1_contrato": "sí" if premium * 100 <= budget else "NO",
        "IV": iv, "costo_lado": slip,
    }
    for y in years[-2:]:
        row[f"avg_r_{y}"] = _year_avg(opt_trades, y)
    return row, opt_trades


def compare(symbols: list[str], data_dir: str | Path = "data", strategy: str = "karen_pullback",
            params: dict | None = None, dte: int = 2, delta: float = 0.70, risk: float = 150,
            account: float = 3000, max_position_pct: float = 30, iv_override: float | None = None,
            slip_override: float | None = None) -> tuple[pd.DataFrame, list[str]]:
    rows, all_trades, missing = [], [], []
    for symbol in symbols:
        path = Path(data_dir) / f"{symbol.upper()}_1min.csv"
        if not path.exists():
            missing.append(symbol)
            continue
        contexts = bt.build_contexts(bt.load_bars_csv(path))
        row, trades = evaluate_symbol(symbol.upper(), contexts, strategy, params or {}, dte, delta,
                                      iv_override or default_iv(symbol), slip_override or default_slip(symbol),
                                      risk, account, max_position_pct)
        rows.append(row)
        all_trades += trades
    if all_trades:
        total = bt.summarize(all_trades)
        rows.append({"activo": "TODOS", "trades": total["trades"], "opc_avg_r": total["avg_r"],
                     f"$_por_trade_{risk:.0f}": round(total["avg_r"] * risk, 2),
                     "opc_PF": total["profit_factor"], "meses_+": total["pct_months_positive"],
                     "peor_racha_r": total["max_drawdown_r"]})
    return pd.DataFrame(rows), missing
