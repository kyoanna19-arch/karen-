from datetime import date, time

import pandas as pd
import pytest

from market_signals import backtest as bt
from market_signals.strategies import DayContext, Signal, gap_fade, orb, allowed_directions
from market_signals.synthetic import synthetic_bars


def make_day(prices, day="2026-03-02", prev_close=100.0, premarket=None):
    """prices: lista de (open, high, low, close) empezando a las 9:30."""
    idx = pd.date_range(f"{day} 09:30", periods=len(prices), freq="1min")
    bars = pd.DataFrame(prices, columns=["open", "high", "low", "close"], index=idx)
    bars["volume"] = 1000.0
    pm = premarket if premarket is not None else bars.iloc[0:0]
    return DayContext(day=date.fromisoformat(day), bars=bars, premarket=pm,
                      prev_close=prev_close, prev_high=prev_close + 1, prev_low=prev_close - 1)


def flat(n, p=100.0):
    return [(p, p + 0.1, p - 0.1, p)] * n


def test_orb_long_hits_target():
    # rango 5m: 99.9-100.1; ruptura a las 9:36, luego sube
    prices = flat(5) + [(100, 100.1, 99.95, 100.05), (100.05, 100.3, 100.0, 100.25),
                        (100.25, 100.3, 100.2, 100.28)] + [(100.3, 101.0, 100.25, 100.9)] * 5 + flat(10, 100.9)
    ctx = make_day(prices, prev_close=99.5)
    sig = orb(ctx, range_minutes=5, target_r=1.0)
    assert sig is not None and sig.direction == 1 and sig.stop == pytest.approx(99.9)
    trade = bt.simulate(ctx, sig, "orb", cost_r=0.1)
    assert trade.exit_reason == "objetivo"
    assert trade.r == pytest.approx(1.0)
    assert trade.r_net == pytest.approx(0.9)


def test_stop_checked_before_target_in_same_bar():
    prices = flat(3) + [(100, 105, 95, 100)] + flat(5)
    ctx = make_day(prices)
    trade = bt.simulate(ctx, Signal(pos=2, direction=1, stop=99, target_r=1), "x", cost_r=0)
    assert trade.exit_reason == "stop" and trade.r == pytest.approx(-1)


def test_time_exit():
    prices = flat(10)
    ctx = make_day(prices)
    trade = bt.simulate(ctx, Signal(pos=0, direction=-1, stop=101, target_r=2, exit_by=time(9, 35)), "x", 0)
    assert trade.exit_reason == "tiempo" and trade.exit_time == time(9, 35)


def test_entry_skipped_when_gapping_through_stop():
    prices = flat(2) + [(98, 98.1, 97.9, 98)] + flat(3, 98)
    ctx = make_day(prices)
    assert bt.simulate(ctx, Signal(pos=1, direction=1, stop=99, target_r=1), "x") is None


def test_direction_filter():
    assert allowed_directions(0.5, "with_gap") == {1}
    assert allowed_directions(-0.5, "against_gap") == {1}
    assert allowed_directions(0.0, "with_gap") == set()
    assert allowed_directions(0.3, "both") == {1, -1}


def test_gap_fade_targets_previous_close():
    # abre 100 con cierre previo 99.5 (gap +0.5%), rompe el rango hacia abajo
    prices = flat(5) + [(100, 100.0, 99.7, 99.75)] + [(99.75, 99.8, 99.4, 99.45)] * 5
    ctx = make_day(prices, prev_close=99.5)
    sig = gap_fade(ctx, min_gap_pct=0.3, range_minutes=5)
    assert sig.direction == -1 and sig.target_price == pytest.approx(99.5)
    trade = bt.simulate(ctx, sig, "gap_fade", cost_r=0)
    assert trade.exit_reason == "objetivo" and trade.exit == pytest.approx(99.5)


def test_full_pipeline_on_synthetic_data(tmp_path):
    bars = synthetic_bars(days=40)
    path = tmp_path / "bars.csv"
    bars.to_csv(path, index_label="datetime")
    contexts = bt.build_contexts(bt.load_bars_csv(path))
    assert len(contexts) == 39
    assert len(contexts[0].premarket) > 0
    ranked = bt.optimize(contexts, ["orb"], min_trades=5)
    assert not ranked.empty
    wf = bt.walk_forward(contexts, names=["orb"], min_trades=5)
    assert list(wf["strategy"]) == ["orb"]
    months = bt.monthly_table(bt.run_strategy(contexts, "orb"))
    assert months["trades"].sum() > 0


def test_no_edge_on_driftless_random_walk():
    """Sin tendencia, ninguna estrategia debería 'ganar' mucho: protege contra ver el futuro."""
    contexts = bt.build_contexts(synthetic_bars(days=300, seed=21))
    for name in ["orb", "premarket_break", "vwap_cross"]:
        stats = bt.summarize(bt.run_strategy(contexts, name, cost_r=0))
        assert stats["avg_r"] < 0.15, (name, stats)
