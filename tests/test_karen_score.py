from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from market_signals import backtest as bt
from market_signals.strategies import DayContext, _last_closed, candles, karen_score
from market_signals.synthetic import synthetic_bars


@pytest.fixture(scope="module")
def contexts():
    return bt.build_contexts(synthetic_bars(days=200, seed=13))


def signals(contexts, **params):
    out = []
    for ctx in contexts:
        sig = karen_score(ctx, **params)
        if sig is not None:
            out.append((ctx, sig))
    return out


def recompute(ctx, sig, ema_sep=0.0003, body_min=0.6):
    """Recalcula gates y condiciones de forma independiente, a partir de los tiempos de las velas."""
    t = ctx.bars.index[sig.pos]
    k1, k5, k15 = candles(ctx, 1), candles(ctx, 5), candles(ctx, 15)
    all_idx = list(ctx.premarket.index) + list(ctx.bars.index)

    def closed(k, minutes):
        # última vela cuyo último minuto es <= t
        ends = [all_idx[p + len(ctx.premarket)] for p in k["end_pos"]]
        js = [j for j, e in enumerate(ends) if e <= t]
        return js[-1]

    a, b, c = closed(k1, 1), closed(k5, 5), closed(k15, 15)
    up = sig.direction == 1
    s = 1 if up else -1
    price = ctx.c[sig.pos]
    gt = lambda x, y: x > y if up else x < y  # noqa: E731
    gates = [(k15["rsi"][c] >= 60) if up else (k15["rsi"][c] <= 40), 25 <= k1["rsi"][a] <= 75,
             gt(k15["ema9"][c], k15["ema21"][c])]
    o5, h5, l5, c5 = (k5[x][b] for x in ("open", "high", "low", "close"))
    conds = {
        "C1": gt(k15["ema9"][c], k15["ema21"][c]), "C2": gt(k5["ema9"][b], k5["ema21"][b]),
        "C3": gt(k1["ema9"][a], k1["ema21"][a]), "C4": (k5["rsi"][b] >= 60) if up else (k5["rsi"][b] <= 40),
        "C5": gt(price, k15["vwap"][c]), "C6": gt(price, k5["vwap"][b]),
        "C7": k5["volume"][b] > np.mean(k5["volume"][b - 4:b]),
        "C8": abs(c5 - o5) / (h5 - l5) >= body_min and gt(c5, o5),
        "C9": (k15["rsi"][c] >= 65) if up else (k15["rsi"][c] <= 35),
        "C10": s * (k5["ema9"][b] - k5["ema21"][b]) / price > ema_sep,
    }
    return gates, {k for k, v in conds.items() if v}, (a, b, c)


@pytest.mark.parametrize("params", [{"min_score": 5}, {"min_score": 7, "sides": "both"}])
def test_every_signal_passes_gates_and_score(contexts, params):
    found = signals(contexts, **params)
    assert len(found) >= 5
    for ctx, sig in found:
        gates, met, (a, b, c) = recompute(ctx, sig)
        assert all(gates)
        assert set(sig.conditions.split(",")) == met
        assert len(met) >= params["min_score"]
        assert "C1" in met  # C1 repite el gate G3: siempre se cumple
        k5 = candles(ctx, 5)
        assert sig.stop == pytest.approx(k5["low"][b] if sig.direction == 1 else k5["high"][b])
        assert ctx.times[sig.pos] <= datetime.strptime("11:55", "%H:%M").time()


def test_calls_only_by_default(contexts):
    assert all(sig.direction == 1 for _, sig in signals(contexts, min_score=5))


def test_higher_score_is_stricter(contexts):
    n = [len(signals(contexts, min_score=m)) for m in (5, 7, 9)]
    assert n[0] >= n[1] >= n[2]


def test_uses_only_closed_candles(contexts):
    ctx = contexts[0]
    k5 = candles(ctx, 5)
    pos = ctx.first_pos_at_or_after(datetime.strptime("10:02", "%H:%M").time())
    j = _last_closed(k5, pos)
    assert ctx.times[int(k5["end_pos"][j])].strftime("%H:%M") == "09:59"


def test_no_lookahead(contexts):
    checked = 0
    for ctx, sig in signals(contexts, min_score=6)[:25]:
        partial = DayContext(day=ctx.day, bars=ctx.bars.iloc[:sig.pos + 2], premarket=ctx.premarket,
                             prev_close=ctx.prev_close, prev_high=ctx.prev_high, prev_low=ctx.prev_low)
        again = karen_score(partial, min_score=6)
        assert again is not None and again.pos == sig.pos and again.conditions == sig.conditions
        checked += 1
    assert checked > 0


def test_condition_impact_report(contexts):
    trades = bt.run_strategy(contexts, "karen_score", {"min_score": 5}, cost_r=0.1)
    table = bt.condition_impact(trades)
    assert list(table["condición"])[:3] == ["C1", "C2", "C3"]
    row = table.set_index("condición").loc["C1"]
    assert row["trades_sí"] == len(trades) and row["trades_no"] == 0


def test_paper_matches_backtest(contexts, tmp_path):
    from tests.test_paper import run_day
    compared = 0
    days = [i for i, ctx in enumerate(contexts) if karen_score(ctx, min_score=5)][:15] + [0, 1, 2]
    for i in days:
        ctx = contexts[i]
        expected = bt.run_strategy([ctx], "karen_score", {"min_score": 5}, cost_r=0)
        _, _, trade = run_day(contexts, i, tmp_path, strategy="karen_score", params={"min_score": 5})
        assert bool(expected) == bool(trade)
        if expected:
            assert trade.exit_reason == expected[0].exit_reason
            compared += 1
    assert compared >= 3
