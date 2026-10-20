from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from market_signals import swing
from market_signals import swing_portfolio as sp


def make_daily(seed, n=1600, drift=0.0005, start_price=100.0):
    rng = np.random.default_rng(seed)
    close = start_price * np.cumprod(1 + rng.normal(drift, 0.012, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


@pytest.fixture(scope="module")
def ind():
    return {s: swing.indicators(make_daily(i, start_price=p)) for i, (s, p) in
            enumerate([("SPY", 300), ("QQQ", 250), ("IWM", 150), ("DIA", 280)])}


def test_single_symbol_matches_swing_backtest(ind):
    rule = swing.SwingRule()
    cfg = sp.PortfolioConfig(rule=rule, max_positions=1, commission=0.0, whole_shares=False)
    state, _ = sp.run_backtest({"SPY": ind["SPY"]}, cfg, 3000)
    raw = make_daily(0, start_price=300)
    trades, _ = swing.backtest(raw, rule, swing.cost_for("SPY"))
    assert len(state.closed) == len(trades) > 10
    for a, b in zip(state.closed, trades):
        assert a["entry_date"] == str(b.entry_date.date()) and a["exit_date"] == str(b.exit_date.date())
        assert a["ret"] == pytest.approx(b.ret, abs=1e-4)


def test_daily_scan_with_saved_state_equals_backtest(ind, tmp_path):
    cfg = sp.PortfolioConfig(max_positions=2)
    full, equity = sp.run_backtest(ind, cfg, 3000)
    dates = equity.index
    path = tmp_path / "state.json"
    state = sp.PortfolioState(cash=3000)
    sp.save_state(state, path)
    for d in dates:  # un día a la vez, guardando y leyendo el archivo como en la vida real
        state = sp.load_state(path, 3000)
        sp.step(state, d, ind, cfg)
        sp.save_state(state, path)
    final = sp.load_state(path, 3000)
    assert final.closed == full.closed
    assert final.cash == pytest.approx(full.cash)
    assert {k: v.qty for k, v in final.positions.items()} == {k: v.qty for k, v in full.positions.items()}


def test_whole_shares_slots_and_cash(ind):
    cfg = sp.PortfolioConfig(max_positions=2, commission=1.0)
    state = sp.PortfolioState(cash=3000)
    max_open = 0
    for d in sorted(set().union(*[set(df.dropna(subset=["sma200"]).index) for df in ind.values()])):
        sp.step(state, d, ind, cfg)
        max_open = max(max_open, len(state.positions))
        assert state.cash >= -1e-6
        assert all(float(p.qty).is_integer() for p in state.positions.values())
    assert max_open <= 2 and len(state.closed) > 20
    assert all(float(t["qty"]).is_integer() for t in state.closed)


def test_entries_ranked_by_deepest_dip(ind):
    # forzar un día donde varios dan señal: el de RSI2 más bajo va primero
    cfg = sp.PortfolioConfig(max_positions=1)
    for d in ind["SPY"].dropna(subset=["sma200"]).index:
        rows = {s: df.loc[d] for s, df in ind.items() if d in df.index}
        sig = [(r["rsi2"], s) for s, r in rows.items() if swing.entry_signal(r, cfg.rule)]
        if len(sig) >= 2:
            state = sp.PortfolioState(cash=3000)
            res = sp.step(state, d, ind, cfg)
            assert [o.symbol for o in res["orders"]] == [min(sig)[1]]
            return
    pytest.skip("no hubo días con varias señales")


def test_scan_first_run_only_processes_last_day(ind):
    state = sp.PortfolioState(cash=3000)
    results = sp.scan(state, ind, sp.PortfolioConfig())
    assert len(results) == 1 and state.last_date == str(ind["SPY"].index[-1].date())
    assert sp.scan(state, ind, sp.PortfolioConfig()) == []  # nada nuevo


def test_cli_scan_end_to_end(tmp_path, monkeypatch, capsys):
    from market_signals.__main__ import main
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    for i, s in enumerate(["SPY", "QQQ"]):
        make_daily(10 + i).to_csv(swing.daily_path(s, tmp_path / "data"), index_label="date")
    main(["swing-scan", "--symbols", "SPY,QQQ", "--no-download"])
    out = capsys.readouterr().out
    assert "SWING" in out and "Capital: $3,000.00" in out
    assert (tmp_path / "data" / "swing_state.json").exists()
    main(["swing-portfolio", "--symbols", "SPY,QQQ"])
    out = capsys.readouterr().out
    assert "posiciones_máx" in out and "Año por año" in out


def test_cli_scan_sends_moo_orders_to_ibkr(tmp_path, monkeypatch, capsys):
    import market_signals.__main__ as cli
    sent = []

    class FakeIB:
        def place_moo(self, symbol, side, qty):
            sent.append((symbol, side, qty))

        def stock_positions(self):
            return {}

        def disconnect(self):
            pass

    monkeypatch.setattr(cli, "_ibkr", lambda client_id=None: FakeIB())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = make_daily(3, n=600)
    # forzar una señal de compra el último día: caída fuerte en tendencia alcista
    df.iloc[-3:, df.columns.get_loc("close")] = df["close"].iloc[-4] * np.array([0.985, 0.97, 0.955])
    df.iloc[-3:, df.columns.get_loc("low")] = df["close"].iloc[-3:] * 0.995
    df.to_csv(swing.daily_path("SPY", tmp_path / "data"), index_label="date")
    ind = swing.indicators(df)
    assert swing.entry_signal(ind.iloc[-1], swing.SwingRule())
    cli.main(["swing-scan", "--symbols", "SPY", "--no-download", "--execute"])
    out = capsys.readouterr().out
    assert sent and sent[0][0] == "SPY" and sent[0][1] == "BUY" and sent[0][2] >= 1
    assert "Orden MOO enviada a IBKR paper" in out and "Mañana en la apertura: COMPRAR" in out
