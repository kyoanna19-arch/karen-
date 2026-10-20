import numpy as np
import pandas as pd
import pytest

from market_signals import swing


def daily(prices):
    idx = pd.bdate_range("2020-01-01", periods=len(prices))
    p = np.asarray(prices, float)
    return pd.DataFrame({"open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 1e6}, index=idx)


def test_enters_after_dip_in_uptrend_and_exits_next_open():
    prices = list(np.linspace(100, 150, 250)) + [140, 132, 126] + [130, 135, 140, 145] + [146] * 20
    df = daily(prices)
    trades, equity = swing.backtest(df, swing.SwingRule("rsi2", 10, "sma5", 10), cost=0)
    assert len(trades) >= 1
    t = trades[0]
    signal_day = df.index.get_loc(t.entry_date) - 1
    assert df["close"].iloc[signal_day] < df["close"].iloc[signal_day - 1]  # entra después de una caída
    assert t.entry == df.loc[t.entry_date, "open"]  # en la apertura siguiente
    assert t.ret == pytest.approx(t.exit / t.entry - 1)
    assert equity.iloc[0] == 1.0


def test_trend_filter_blocks_downtrend():
    prices = list(np.linspace(200, 100, 300))
    df = daily(prices)
    with_filter, _ = swing.backtest(df, swing.SwingRule("rsi2", 10, "sma5", 10, True))
    without, _ = swing.backtest(df, swing.SwingRule("rsi2", 30, "sma5", 10, False))
    assert with_filter == [] and len(without) >= 0


def test_max_days_and_costs():
    rng = np.random.default_rng(1)
    prices = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 1500))
    df = daily(prices)
    rule = swing.SwingRule("rsi2", 15, "rsi70", 3)
    trades, equity = swing.backtest(df, rule, cost=0.0)
    costly, _ = swing.backtest(df, rule, cost=0.001)
    assert trades and all(t.days <= 4 for t in trades)
    assert np.mean([t.ret for t in costly]) < np.mean([t.ret for t in trades])
    st = swing.stats(trades, equity, df)
    assert st["trades"] == len(trades) and -100 < st["caída_máx_%"] <= 0
    assert set(swing.yearly(trades).columns) == {"trades", "ganadores", "resultado_%"}


def test_compare_and_grid(tmp_path):
    rng = np.random.default_rng(2)
    for sym in ("SPY", "NVDA"):
        prices = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 1200))
        daily(prices).to_csv(swing.daily_path(sym, tmp_path), index_label="date")
    table, missing = swing.compare_symbols(["SPY", "NVDA", "AMD"], folder=tmp_path)
    assert missing == ["AMD"] and list(table["activo"]) == ["SPY", "NVDA"]
    grid = swing.grid_table(swing.load_daily(swing.daily_path("SPY", tmp_path)), "SPY")
    assert len(grid) == len(swing.GRID)
    assert swing.cost_for("SPY") < swing.cost_for("NVDA")
    assert swing.shares_for(3000, 650) == 4
