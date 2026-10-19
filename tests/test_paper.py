from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from market_signals import backtest as bt
from market_signals.config import Settings
from market_signals.ibkr import IBKRError, bars_to_frame, check_paper, round_tick
from market_signals.paper import PaperTrader, journal_summary
from market_signals.synthetic import synthetic_bars

SETTINGS = Settings("t", "sandbox", "", "", "", 8000, 1.0, 30)            # modo prima (escalones)
SETTINGS_LEVELS = Settings("t", "sandbox", "", "", "", 100000, 1.0, 30)   # modo niveles: siempre alcanza


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

    def option_candidates(self, symbol, direction, price, today, min_dte=0):
        self.requested_dte = min_dte
        return [{"symbol": f"{symbol}-OPT", "underlying": symbol, "option_type": "call" if direction == 1 else "put",
                 "strike": round(price), "expiration": str(today), "bid": 1.18, "ask": 1.20,
                 "greeks": {"delta": 0.45 * direction}}]


class FakeBroker:
    def __init__(self, feed, stop_hits=False, price_stops=False):
        self.feed, self.stop_hits, self.price_stops = feed, stop_hits, price_stops
        self.orders = []
        self.entry_underlying = None
        self.direction = 1

    def buy(self, option, qty, limit):
        self.orders.append(("BUY", option["symbol"], qty, limit))
        self.entry_underlying = self.feed.last_price
        self.direction = 1 if option["option_type"] == "call" else -1
        return qty, limit

    def option_price(self):
        move = (self.feed.last_price - self.entry_underlying) * self.direction
        return max(0.01, 1.20 + 0.45 * move)

    def protective_stop(self, option, qty, stop_price):
        self.orders.append(("STOP", option["symbol"], qty, round(stop_price, 2)))
        return SimpleNamespace(kind="stop", price=stop_price, qty=qty, cancelled=False)

    def sell_limit(self, option, qty, price):
        self.orders.append(("LIMIT", option["symbol"], qty, round(price, 2)))
        return SimpleNamespace(kind="limit", price=price, qty=qty, cancelled=False)

    def order_fill(self, handle):
        if handle.cancelled:
            return None
        if handle.kind == "stop":
            hit = self.stop_hits or (self.price_stops and self.option_price() <= handle.price)
            return handle.price if hit else None
        return handle.price if self.option_price() >= handle.price else None

    def cancel(self, handle):
        handle.cancelled = True
        self.orders.append(("CANCEL", handle.kind, handle.qty))

    def sell(self, option, qty, open_orders):
        for h in open_orders:
            self.cancel(h)
        self.orders.append(("SELL", option["symbol"], qty))
        return self.option_price()


def day_setup(contexts, i):
    ctx = contexts[i]
    bars = pd.concat([ctx.premarket, ctx.bars])
    prev = {"close": ctx.prev_close, "high": ctx.prev_high, "low": ctx.prev_low}
    return ctx, FakeFeed(bars, prev)


@pytest.fixture(scope="module")
def contexts():
    return bt.build_contexts(synthetic_bars(days=25, seed=9))


def run_day(contexts, i, tmp_path, stop_hits=False, strategy="orb", params=None, plan=None):
    ctx, feed = day_setup(contexts, i)
    broker = FakeBroker(feed, stop_hits, price_stops=plan is not None)
    clock = Clock(datetime.combine(ctx.day, datetime.min.time()).replace(hour=9, minute=30, second=5))
    settings = SETTINGS if plan is not None else SETTINGS_LEVELS
    trader = PaperTrader(feed, broker, settings, lambda m: None, "SPY", strategy, params or {},
                         now=clock.now, sleep=clock.sleep, poll_seconds=60, journal=tmp_path / "j.csv",
                         exit_plan=plan)
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
        assert [o[0] for o in broker.orders] in (["BUY", "STOP", "CANCEL", "SELL"], ["BUY", "STOP"])
        if trade.exit_reason in ("stop", "objetivo"):
            assert trade.r_underlying == pytest.approx(expected[0].r, abs=0.02)
        compared += 1
    assert compared >= 10


def test_position_size_by_etf_stop_and_cash_cap(contexts, tmp_path):
    ctx, feed = day_setup(contexts, 0)

    def make(settings):
        broker = FakeBroker(feed)
        clock = Clock(datetime.combine(ctx.day, datetime.min.time()).replace(hour=9, minute=30, second=5))
        trader = PaperTrader(feed, broker, settings, lambda m: None, "SPY", "orb", {}, now=clock.now,
                             sleep=clock.sleep, poll_seconds=60, journal=tmp_path / "j.csv")
        return trader, broker, trader.run()

    trader, broker, trade = make(Settings("t", "sandbox", "", "", "", 8000, 5.0, 30))
    risk_u = abs(trader.entry_price - trader.sig.stop)
    loss_per_contract = (0.45 * risk_u + 0.02) * 100  # delta x distancia al stop + spread
    expected = min(int(400 // loss_per_contract), int(8000 * 0.30 // 120))
    assert trade.qty == expected and broker.orders[0] == ("BUY", "SPY-OPT", expected, 1.20)
    assert broker.orders[1] == ("STOP", "SPY-OPT", expected, 0.84)  # red de seguridad: -30% de la prima

    # cuenta chica: el tope de 30% manda (30% de $500 = $150 -> 1 contrato de $120)
    _, _, small = make(Settings("t", "sandbox", "", "", "", 500, 50.0, 30, 30.0))
    assert small.qty == 1


def test_protective_stop_fill_closes_trade_without_selling(contexts, tmp_path):
    _, broker, trade = run_day(contexts, 0, tmp_path, stop_hits=True)
    assert trade.exit_reason == "stop opción" and trade.option_exit == pytest.approx(0.84)
    assert trade.pnl == pytest.approx((0.84 - 1.20) * 100 * trade.qty)
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
    feed.option_candidates = lambda *a, **k: [{"symbol": "X", "underlying": "SPY", "option_type": "call",
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


def test_split_qty():
    from market_signals.paper import split_qty
    half = ((0.25, 0.5), (0.5, 0.5))
    assert split_qty(2, half) == [(1, 0.25), (1, 0.5)]
    assert split_qty(1, half) == [(1, 0.25), (0, 0.5)]
    assert split_qty(5, half) == [(2, 0.25), (3, 0.5)]
    assert split_qty(4, ((0.3, 0.5),)) == [(2, 0.3)]  # el resto corre hasta la hora límite
    assert split_qty(3, ((0.25, 0.34), (0.5, 0.33), (1.0, 0.33))) == [(1, 0.25), (1, 0.5), (1, 1.0)]


def test_premium_plan_scales_out_and_moves_stop(contexts, tmp_path):
    from market_signals.options_sim import ExitPlan
    plan = ExitPlan(stop_pct=0.25, tiers=((0.25, 0.5), (0.5, 0.5)), breakeven=True)
    seen = set()
    for i in range(len(contexts)):
        _, broker, trade = run_day(contexts, i, tmp_path, plan=plan)
        if trade is None:
            continue
        kinds = [o[0] for o in broker.orders]
        assert kinds[:4] == ["BUY", "STOP", "LIMIT", "LIMIT"]
        assert broker.orders[1][3] == pytest.approx(0.90)  # -25% de $1.20
        assert broker.orders[2][3] == pytest.approx(1.50) and broker.orders[3][3] == pytest.approx(1.80)
        reasons = trade.exit_reason.split(" / ")
        seen.add(trade.exit_reason)
        if reasons[0] == "+25%":
            # después del primer escalón el stop se rehace a breakeven por 1 contrato
            assert ("STOP", "SPY-OPT", 1, 1.2) in broker.orders
            assert reasons[1] in ("+50%", "breakeven", "tiempo")
        elif reasons == ["stop opción"]:
            assert trade.pnl == pytest.approx((0.90 - 1.20) * 100 * 2)
        expected_pnl = {"+25%": 0.30, "+50%": 0.60, "breakeven": 0.0, "stop opción": -0.30}
        if len(reasons) == 2 and all(r in expected_pnl for r in reasons):  # 1 contrato por escalón
            assert trade.pnl == pytest.approx(sum(expected_pnl[r] * 100 for r in reasons))
        assert trade.qty == 2
    assert any(r.startswith("+25%") for r in seen) and "stop opción" in seen


def test_paper_time_stop_and_option_choice(contexts, tmp_path):
    from market_signals.options_sim import ExitPlan
    plan = ExitPlan(stop_pct=0.25, tiers=((5.0, 1.0),), breakeven=False, max_minutes=10)  # escalón imposible
    ctx, feed = day_setup(contexts, 0)
    broker = FakeBroker(feed, price_stops=False)
    clock = Clock(datetime.combine(ctx.day, datetime.min.time()).replace(hour=9, minute=30, second=5))
    trader = PaperTrader(feed, broker, SETTINGS, lambda m: None, "SPY", "orb", {}, now=clock.now,
                         sleep=clock.sleep, poll_seconds=60, journal=tmp_path / "j.csv", exit_plan=plan,
                         min_dte=2, target_delta=0.65)
    trade = trader.run()
    assert feed.requested_dte == 2
    assert trade.exit_reason == "no funcionó en 10 min"
    entry = datetime.strptime(trade.signal_time, "%H:%M")
    exit_ = datetime.strptime(trade.exit_time, "%H:%M")
    assert 10 <= (exit_ - entry).seconds / 60 <= 12
