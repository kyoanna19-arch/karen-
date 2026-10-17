from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from market_signals import backtest as bt
from market_signals.config import Settings
from market_signals.ibkr import IBKRError, bars_to_frame, check_paper, round_tick
from market_signals.paper import PaperTrader, journal_summary
from market_signals.synthetic import synthetic_bars

SETTINGS = Settings("t", "sandbox", "", "", "", 8000, 1.0, 30)


class Clock:
    def __init__(self, start):
        self.t = start

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += timedelta(seconds=seconds)


class FakeFeed:
    """Reproduce un día histórico minuto a minuto, como si fuera en vivo."""

    def __init__(self, day_bars, prev):
        self.day_bars, self.prev = day_bars, prev
        self.last_price = None

    def minute_bars(self, symbol, now):
        bars = self.day_bars[self.day_bars.index <= now]  # incluye la vela en formación
        self.last_price = float(bars["close"].iloc[-1])
        return bars

    def prev_day(self, symbol, today):
        return self.prev

    def option_candidates(self, symbol, direction, price, today):
        return [{"symbol": f"{symbol}-OPT", "underlying": symbol, "option_type": "call" if direction == 1 else "put",
                 "strike": round(price), "expiration": str(today), "bid": 1.18, "ask": 1.20,
                 "greeks": {"delta": 0.45 * direction}}]


class FakeBroker:
    def __init__(self, feed, stop_hits=False):
        self.feed, self.stop_hits = feed, stop_hits
        self.orders = []
        self.entry_underlying = None
        self.direction = 1

    def buy(self, option, qty, limit):
        self.orders.append(("BUY", option["symbol"], qty, limit))
        self.entry_underlying = self.feed.last_price
        self.direction = 1 if option["option_type"] == "call" else -1
        return qty, limit

    def protective_stop(self, option, qty, stop_price):
        self.orders.append(("STOP", option["symbol"], qty, round(stop_price, 2)))
        return SimpleNamespace(price=stop_price)

    def stop_fill(self, handle):
        return handle.price if self.stop_hits else None

    def sell(self, option, qty, stop_handle):
        self.orders.append(("SELL", option["symbol"], qty))
        move = (self.feed.last_price - self.entry_underlying) * self.direction
        return max(0.01, 1.20 + 0.45 * move)


def day_setup(contexts, i):
    ctx = contexts[i]
    bars = pd.concat([ctx.premarket, ctx.bars])
    prev = {"close": ctx.prev_close, "high": ctx.prev_high, "low": ctx.prev_low}
    return ctx, FakeFeed(bars, prev)


@pytest.fixture(scope="module")
def contexts():
    return bt.build_contexts(synthetic_bars(days=25, seed=9))


def run_day(contexts, i, tmp_path, stop_hits=False, strategy="orb", params=None):
    ctx, feed = day_setup(contexts, i)
    broker = FakeBroker(feed, stop_hits)
    clock = Clock(datetime.combine(ctx.day, datetime.min.time()).replace(hour=9, minute=30, second=5))
    trader = PaperTrader(feed, broker, SETTINGS, lambda m: None, "SPY", strategy, params or {},
                         now=clock.now, sleep=clock.sleep, poll_seconds=60, journal=tmp_path / "j.csv")
    return ctx, broker, trader.run()


def test_paper_matches_backtest_exit_reasons(contexts, tmp_path):
    """El paper debe salir por el mismo motivo que el backtest en el mismo día."""
    compared = 0
    for i in range(len(contexts)):
        ctx, broker, trade = run_day(contexts, i, tmp_path)
        expected = bt.run_strategy([ctx], "orb", {}, cost_r=0)
        if not expected:
            assert trade is None
            continue
        assert trade is not None
        assert trade.exit_reason == expected[0].exit_reason
        signal_bar = ctx.times[ctx.times.index(expected[0].entry_time) - 1]  # backtest entra en la vela siguiente
        assert trade.signal_time == f"{signal_bar:%H:%M}"
        assert [o[0] for o in broker.orders] == ["BUY", "STOP", "SELL"]
        if trade.exit_reason in ("stop", "objetivo"):
            assert trade.r_underlying == pytest.approx(expected[0].r, abs=0.02)
        compared += 1
    assert compared >= 10


def test_position_size_and_protective_stop(contexts, tmp_path):
    _, broker, trade = run_day(contexts, 0, tmp_path)
    # $8000 x 1% = $80; prima $1.20 x 100 x 30% = $36 -> 2 contratos; stop en $0.84
    assert broker.orders[0] == ("BUY", "SPY-OPT", 2, 1.20)
    assert broker.orders[1] == ("STOP", "SPY-OPT", 2, 0.84)
    assert trade.qty == 2


def test_protective_stop_fill_closes_trade_without_selling(contexts, tmp_path):
    _, broker, trade = run_day(contexts, 0, tmp_path, stop_hits=True)
    assert trade.exit_reason == "stop opción" and trade.option_exit == pytest.approx(0.84)
    assert trade.pnl == pytest.approx((0.84 - 1.20) * 200)
    assert "SELL" not in [o[0] for o in broker.orders]


def test_journal_written_and_summarized(contexts, tmp_path):
    for i in range(6):
        run_day(contexts, i, tmp_path)
    df = pd.read_csv(tmp_path / "j.csv")
    assert len(df) >= 3 and set(df["exit_reason"]) <= {"stop", "objetivo", "tiempo"}
    text = journal_summary(tmp_path / "j.csv")
    assert f"Trades: {len(df)}" in text and "Por mes:" in text


def test_expensive_premium_skips_trade(contexts, tmp_path):
    ctx, feed = day_setup(contexts, 0)
    feed.option_candidates = lambda *a: [{"symbol": "X", "underlying": "SPY", "option_type": "call",
                                          "strike": 1, "expiration": "x", "bid": 9.9, "ask": 10.0, "greeks": None},
                                         {"symbol": "Y", "underlying": "SPY", "option_type": "put",
                                          "strike": 1, "expiration": "x", "bid": 9.9, "ask": 10.0, "greeks": None}]
    broker = FakeBroker(feed)
    clock = Clock(datetime.combine(ctx.day, datetime.min.time()).replace(hour=9, minute=30))
    msgs = []
    out = PaperTrader(feed, broker, SETTINGS, msgs.append, "SPY", "orb", {}, now=clock.now, sleep=clock.sleep,
                      poll_seconds=60, journal=tmp_path / "j.csv").run()
    assert out is None and broker.orders == [] and any("muy cara" in m for m in msgs)


def test_no_signal_stops_at_until(contexts, tmp_path):
    ctx, feed = day_setup(contexts, 0)
    clock = Clock(datetime.combine(ctx.day, datetime.min.time()).replace(hour=9, minute=30))
    out = PaperTrader(feed, FakeBroker(feed), SETTINGS, lambda m: None, "SPY", "gap_fade",
                      {"min_gap_pct": 50}, now=clock.now, sleep=clock.sleep, poll_seconds=60,
                      until="10:00", journal=tmp_path / "j.csv").run()
    assert out is None and clock.t.time().hour == 10


def test_only_paper_accounts_allowed():
    check_paper(["DU1234567"], 7497)
    with pytest.raises(IBKRError, match="REAL"):
        check_paper(["DU1234567"], 7496)
    with pytest.raises(IBKRError, match="no son paper"):
        check_paper(["U1234567"], 7497)
    with pytest.raises(IBKRError):
        check_paper([], 4002)


def test_round_tick():
    assert round_tick(5.13, "AAPL") == 5.15
    assert round_tick(5.13, "SPY") == 5.13
    assert round_tick(1.234, "AAPL") == 1.23


def test_ibkr_bars_converted_to_eastern():
    bars = [SimpleNamespace(date=datetime(2026, 3, 4, 14, 30, tzinfo=__import__("datetime").timezone.utc),
                            open=1, high=2, low=0.5, close=1.5, volume=100)]
    df = bars_to_frame(bars)
    assert df.index[0] == pd.Timestamp("2026-03-04 09:30") and df["volume"].iloc[0] == 100.0
