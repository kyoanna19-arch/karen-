import pandas as pd
import pytest

from market_signals import backtest as bt
from market_signals.signal_study import _forward, baseline_returns, signal_returns, study, summarize_returns
from market_signals.synthetic import synthetic_bars


@pytest.fixture(scope="module")
def contexts():
    return bt.build_contexts(synthetic_bars(days=60, seed=8))


def test_forward_returns_math(contexts):
    ctx = contexts[3]
    e = 20
    out = _forward(contexts, 3, e, 1)
    entry = ctx.o[e]
    assert out["5min"] == pytest.approx((ctx.c[e + 4] / entry - 1) * 100)
    assert out["cierre"] == pytest.approx((ctx.c[-1] / entry - 1) * 100)
    assert out["+1día"] == pytest.approx((contexts[4].c[-1] / entry - 1) * 100)
    short = _forward(contexts, 3, e, -1)
    assert short["5min"] == pytest.approx(-out["5min"])
    assert "+5días" not in _forward(contexts, len(contexts) - 2, e, 1)


def test_signal_and_baseline(contexts):
    sig = signal_returns(contexts, "orb", {})
    base = baseline_returns(contexts)
    assert len(sig) > 10 and len(base) > len(sig)
    summary = summarize_returns(sig)
    assert summary["5min"]["n"] == sig["5min"].notna().sum()


def test_study_tables(tmp_path):
    for i, sym in enumerate(["SPY", "QQQ"]):
        synthetic_bars(days=50, seed=20 + i).to_csv(tmp_path / f"{sym}_1min.csv", index_label="datetime")
    per_symbol, pooled, missing = study(["SPY", "QQQ", "AMD"], "orb", {}, data_dir=tmp_path)
    assert missing == ["AMD"] and list(per_symbol["activo"]) == ["SPY", "QQQ"]
    row = pooled.set_index("plazo").loc["15min"]
    assert row["ventaja_%"] == pytest.approx(row["señal_prom_%"] - row["azar_prom_%"], abs=2e-3)
