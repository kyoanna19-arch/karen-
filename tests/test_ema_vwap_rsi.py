from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_signals import backtest as bt
from market_signals.strategies import DayContext, candles, ema_vwap_rsi, rsi
from market_signals.synthetic import synthetic_bars


@pytest.fixture(scope="module")
def contexts():
    return bt.build_contexts(synthetic_bars(days=120, seed=5))


def test_rsi_bounds_and_direction():
    up = rsi(np.linspace(100, 110, 50))
    down = rsi(np.linspace(110, 100, 50))
    assert up[-1] > 95 and down[-1] < 5
    assert np.all((up >= 0) & (up <= 100))


def test_five_minute_candles_aggregate_correctly(contexts):
    ctx = contexts[0]
    k = candles(ctx, 5)
    j = int(np.argmax(k["regular"]))  # primera vela regular: 9:30-9:34
    first5 = ctx.bars.iloc[:5]
    assert k["open"][j] == first5["open"].iloc[0]
    assert k["high"][j] == first5["high"].max()
    assert k["low"][j] == first5["low"].min()
    assert k["close"][j] == first5["close"].iloc[-1]
    assert k["volume"][j] == first5["volume"].sum()
    assert k["end_pos"][j] == 4 and ctx.times[4].strftime("%H:%M") == "09:34"
    assert not k["regular"][:j].any() and k["regular"][j:].all()


@pytest.mark.parametrize("tf", [1, 2, 5])
def test_every_signal_satisfies_all_rules(contexts, tf):
    params = dict(timeframe=tf, rsi_call=55, rsi_put=45)
    found = 0
    for ctx in contexts:
        sig = ema_vwap_rsi(ctx, **params)
        if sig is None:
            continue
        found += 1
        k = candles(ctx, tf)
        j = int(np.where(k["end_pos"] == sig.pos)[0][-1])
        o, h, l, c, v = (k[x][j] for x in ("open", "high", "low", "close", "volume"))
        e9, e21, r, vw = k["ema9"], k["ema21"], k["rsi"][j], k["vwap"]
        assert abs(c - o) / (h - l) >= 0.6                      # vela sólida
        assert v > k["volume"][j - 1]                           # volumen
        if sig.direction == 1:
            assert e9[j] > e21[j] and e9[j] > e9[j - 1] and e21[j] > e21[j - 1]
            assert l <= e9[j] and c > max(e9[j], e21[j])        # rompe los promedios
            assert c > vw[j] and vw[j] > vw[j - 3]              # sobre VWAP girado
            assert r > 55 and sig.stop == l
        else:
            assert e9[j] < e21[j] and e9[j] < e9[j - 1] and e21[j] < e21[j - 1]
            assert h >= e9[j] and c < min(e9[j], e21[j])
            assert c < vw[j] and vw[j] < vw[j - 3]
            assert r < 45 and sig.stop == h
    assert found > 0


def test_live_ignores_incomplete_candle(contexts):
    """Con velas de 5m, a las 9:43 la vela 9:40-9:44 aún no cierra: no puede dar señal."""
    ctx = contexts[3]
    for minutes in range(10, 120):
        partial = DayContext(day=ctx.day, bars=ctx.bars.iloc[:minutes], premarket=ctx.premarket,
                             prev_close=ctx.prev_close, prev_high=ctx.prev_high, prev_low=ctx.prev_low, live=True)
        sig = ema_vwap_rsi(partial, timeframe=5, rsi_call=0, rsi_put=100, body_min=0.3)
        if sig is not None:
            assert partial.times[sig.pos].minute % 5 == 4  # termina en :x4 o :x9


def test_signal_does_not_change_when_future_bars_are_added(contexts):
    """La señal a las 10:00 debe ser la misma sin importar lo que pase después."""
    for ctx in contexts[:40]:
        full = ema_vwap_rsi(ctx, timeframe=2)
        if full is None:
            continue
        cut = full.pos + 2
        partial = DayContext(day=ctx.day, bars=ctx.bars.iloc[:cut], premarket=ctx.premarket,
                             prev_close=ctx.prev_close, prev_high=ctx.prev_high, prev_low=ctx.prev_low)
        again = ema_vwap_rsi(partial, timeframe=2)
        assert again is not None and again.pos == full.pos and again.stop == pytest.approx(full.stop)


def test_ablation_runs(contexts):
    table = bt.ablation(contexts[:60], "ema_vwap_rsi", {"timeframe": 2})
    assert table["variante"].iloc[0] == "completa" and "sin RSI" in set(table["variante"])
    no_rsi = table.set_index("variante").loc["sin RSI", "trades"]
    assert no_rsi >= table.set_index("variante").loc["completa", "trades"]
