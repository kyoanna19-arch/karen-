import pytest

from market_signals import backtest as bt
from market_signals.strategies import DayContext, _last_closed, candles, karen_pullback, karen_setups
from market_signals.synthetic import synthetic_bars

PARAMS = [{"min_score": 7}, {"min_score": 7, "pullback_to": "mid_candle5", "stop_mode": "setup", "wait_minutes": 30}]


@pytest.fixture(scope="module")
def contexts():
    return bt.build_contexts(synthetic_bars(days=200, seed=13))


@pytest.mark.parametrize("params", PARAMS)
def test_entry_comes_after_a_setup_and_a_pullback(contexts, params):
    found = 0
    for ctx in contexts:
        sig = karen_pullback(ctx, **params)
        if sig is None:
            continue
        found += 1
        wait = params.get("wait_minutes", 20)
        k5 = candles(ctx, 5)

        def valid(setup):
            b = _last_closed(k5, setup.pos)
            # dentro de la ventana de espera y sin romper el mínimo de la vela del setup
            return (0 < sig.pos - setup.pos <= wait
                    and ctx.l[setup.pos + 1:sig.pos + 1].min() > k5["low"][b])

        candidates = [s for s in karen_setups(ctx, params["min_score"]) if valid(s)]
        assert candidates, "la entrada debe venir de un setup válido"
        assert ctx.c[sig.pos] > ctx.h[sig.pos - 1]  # retoma la subida
        assert sig.stop < ctx.c[sig.pos] and sig.direction == 1
        if params.get("stop_mode", "pullback") == "setup":
            lows = [k5["low"][_last_closed(k5, c.pos)] for c in candidates]
            assert any(sig.stop == pytest.approx(low) for low in lows)
    assert found >= 5


def test_pullback_no_lookahead(contexts):
    checked = 0
    for ctx in contexts:
        sig = karen_pullback(ctx, min_score=7)
        if sig is None:
            continue
        partial = DayContext(day=ctx.day, bars=ctx.bars.iloc[:sig.pos + 2], premarket=ctx.premarket,
                             prev_close=ctx.prev_close, prev_high=ctx.prev_high, prev_low=ctx.prev_low)
        again = karen_pullback(partial, min_score=7)
        assert again is not None and again.pos == sig.pos and again.stop == pytest.approx(sig.stop)
        checked += 1
        if checked >= 20:
            break
    assert checked > 0


def test_paper_matches_backtest_pullback(contexts, tmp_path):
    from tests.test_paper import run_day
    days = [i for i, ctx in enumerate(contexts) if karen_pullback(ctx, min_score=7)][:10]
    for i in days:
        expected = bt.run_strategy([contexts[i]], "karen_pullback", {"min_score": 7}, cost_r=0)
        _, _, trade = run_day(contexts, i, tmp_path, strategy="karen_pullback", params={"min_score": 7})
        assert bool(expected) == bool(trade)
        if expected:
            assert trade.exit_reason == expected[0].exit_reason
