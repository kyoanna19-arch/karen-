from market_signals.compare import compare, default_iv, default_slip
from market_signals.synthetic import synthetic_bars


def test_compare_table(tmp_path):
    for i, sym in enumerate(["QQQ", "NVDA"]):
        synthetic_bars(days=120, seed=30 + i, price=200 + 300 * i).to_csv(tmp_path / f"{sym}_1min.csv",
                                                                         index_label="datetime")
    table, missing = compare(["QQQ", "NVDA", "AMD"], data_dir=tmp_path, params={"min_score": 7},
                             risk=150, account=3000, max_position_pct=30)
    assert missing == ["AMD"]
    assert list(table["activo"]) == ["QQQ"] * 3 + ["NVDA"] * 3 + ["TODOS"] * 3
    rows = table.set_index(["activo", "estructura"])
    for structure in ("CALL", "spread objetivo", "spread ATM"):
        assert rows.loc[("TODOS", structure), "trades"] == (rows.loc[("QQQ", structure), "trades"]
                                                            + rows.loc[("NVDA", structure), "trades"])
    # el spread cuesta menos que el CALL solo
    for sym in ("QQQ", "NVDA"):
        assert rows.loc[(sym, "spread ATM"), "costo_contrato"] < rows.loc[(sym, "CALL"), "costo_contrato"]
    # NVDA sintético a ~$500: el CALL ITM no cabe en $900
    assert rows.loc[("NVDA", "CALL"), "cabe"] == "NO"
    assert default_slip("spy") == 0.02 and default_iv("XYZ") == 0.35


def test_spread_math():
    from datetime import time
    import pytest
    from market_signals import options_sim as osim
    from market_signals.strategies import Signal
    from tests.test_backtest import make_day
    # sube directo al objetivo: el spread gana; cae al stop: pierde 1R
    up = make_day([(500, 500.05, 499.95, 500)] * 2 + [(500, 503, 500, 502.9)] * 60)
    down = make_day([(500, 500.05, 499.95, 500)] * 2 + [(500, 500, 498, 498.2)] * 60)
    sig = Signal(pos=0, direction=1, stop=499, target_r=2, exit_by=time(12, 0))
    for short in ("target", "atm"):
        win = osim.simulate_spread_levels(up, sig, "x", delta=0.7, dte=2, slip=0, short=short)
        loss = osim.simulate_spread_levels(down, sig, "x", delta=0.7, dte=2, slip=0, short=short)
        assert win.exit_reason == "objetivo" and win.r > 0.5
        assert loss.exit_reason == "stop" and loss.r == pytest.approx(-1.0, abs=0.05)
        assert win.entry < osim.bs_call(500, osim.strike_for_delta(500, osim.years_left(time(9, 31), 2), 0.18, 0.7),
                                         osim.years_left(time(9, 31), 2), 0.18)
    assert osim.strike_step(150, etf=False) == 1.0 and osim.strike_step(300, etf=False) == 2.5
    assert osim.strike_step(700, etf=False) == 5.0
