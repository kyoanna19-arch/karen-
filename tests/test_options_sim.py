from datetime import time

import pytest

from market_signals import options_sim as osim
from market_signals.strategies import Signal
from tests.test_backtest import make_day


def test_black_scholes_basics():
    T = osim.years_left(time(10, 0))
    c, p = osim.bs_call(500, 500, T, 0.18), osim.bs_put(500, 500, T, 0.18)
    assert 1.5 < c < 3.5  # 0DTE ATM de SPY a las 10:00 cuesta del orden de $2
    assert c - p == pytest.approx(500 - 500 * __import__("math").exp(-0.04 * T), abs=1e-6)  # paridad put-call
    assert osim.bs_call(500, 500, osim.years_left(time(15, 30)), 0.18) < c  # theta: vale menos más tarde
    K = osim.strike_for_delta(500, T, 0.18, 0.45)
    assert K > 500 and abs(osim.bs_delta(500, K, T, 0.18) - 0.45) < 0.1


def rising_then(prices_after):
    return make_day([(500, 500.1, 499.9, 500)] * 3 + prices_after + [(p[3], p[3], p[3], p[3]) for p in prices_after[-1:]] * 200)


def test_stop_on_premium_is_minus_one_r():
    ctx = rising_then([(500, 500, 495, 495)])  # SPY cae $5: la opción pierde mucho más de 25%
    plan = osim.ExitPlan(stop_pct=0.25, tiers=((0.5, 1.0),), breakeven=False)
    t = osim.simulate_option(ctx, Signal(pos=2, direction=1, stop=0, exit_by=time(12, 0)), "x", plan, slip=0)
    assert t.exit_reason == "stop" and t.r == pytest.approx(-1.0)


def test_scaled_exit_with_breakeven():
    # sube fuerte (toca +25% pero no +50%) y luego se desploma
    ctx = make_day([(500, 500.1, 499.9, 500)] * 3 + [(500, 501.2, 500, 501.2)] + [(501.2, 501.2, 494, 494)] * 50)
    plan = osim.ExitPlan(stop_pct=0.25, tiers=((0.25, 0.5), (0.5, 0.5)), breakeven=True)
    t = osim.simulate_option(ctx, Signal(pos=2, direction=1, stop=0, exit_by=time(12, 0)), "x", plan, slip=0)
    assert t.exit_reason == "+25% / breakeven"
    assert t.r == pytest.approx(0.5 * 0.25 / 0.25)  # mitad ganó 25%, la otra mitad salió en 0
    no_be = osim.ExitPlan(stop_pct=0.25, tiers=((0.25, 0.5), (0.5, 0.5)), breakeven=False)
    t2 = osim.simulate_option(ctx, Signal(pos=2, direction=1, stop=0, exit_by=time(12, 0)), "x", no_be, slip=0)
    assert t2.exit_reason == "+25% / stop" and t2.r == pytest.approx((0.5 * 0.25 - 0.5 * 0.25) / 0.25)


def test_time_exit_loses_theta_when_flat():
    ctx = make_day([(500, 500.05, 499.95, 500)] * 200)
    plan = osim.ExitPlan(stop_pct=0.5, tiers=((0.5, 1.0),), breakeven=False)
    t = osim.simulate_option(ctx, Signal(pos=0, direction=1, stop=0, exit_by=time(11, 30)), "x", plan, slip=0)
    assert t.exit_reason == "tiempo" and t.r < 0  # precio plano: la opción pierde por tiempo


def test_slippage_costs():
    ctx = make_day([(500, 500.05, 499.95, 500)] * 200)
    plan = osim.ExitPlan(stop_pct=0.5, tiers=((0.5, 1.0),), breakeven=False)
    sig = Signal(pos=0, direction=1, stop=0, exit_by=time(9, 40))
    assert osim.simulate_option(ctx, sig, "x", plan, slip=0.02).r < osim.simulate_option(ctx, sig, "x", plan, slip=0).r


def test_option_path_mae_before_target():
    ctx = make_day([(500, 500.1, 499.9, 500)] * 3 + [(500, 500, 499.4, 499.5)] + [(499.5, 502, 499.5, 502)] + [(502, 502.1, 501.9, 502)] * 50)
    path = osim.option_path(ctx, Signal(pos=2, direction=1, stop=0, exit_by=time(12, 0)), slip=0)
    assert path.mae < -0.1 and path.mfe > 0.5
    assert path.mae_before[0.25] == pytest.approx(path.mae)  # la caída ocurrió antes de ganar


def test_stop_study_and_plans_on_synthetic():
    from market_signals import backtest as bt
    from market_signals.strategies import karen_score
    from market_signals.synthetic import synthetic_bars
    contexts = bt.build_contexts(synthetic_bars(days=120, seed=13))
    pairs = osim.signals_for(contexts, karen_score, {"min_score": 6})
    study = osim.stop_study(pairs)
    assert study["paths"] == len(pairs) and len(study["table"]) == 4
    for row in study["table"]:
        vals = [row[f"sacados_stop_{int(s * 100)}%"] for s in osim.STOP_LEVELS]
        if vals[0] is not None:
            assert vals == sorted(vals, reverse=True)  # stop más amplio saca menos trades buenos
    for plan in osim.EXIT_PLANS:
        trades = [osim.simulate_option(c, s, "k", plan) for c, s in pairs]
        assert all(t is None or t.r >= -1.0 - 1e-9 or "tiempo" in t.exit_reason or "stop" in t.exit_reason
                   for t in trades)
