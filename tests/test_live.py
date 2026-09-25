from datetime import date

import pytest

from market_signals.options import choose_expiration, contracts_for_risk, pick_contract
from market_signals.premarket import bias_label, bias_score
from market_signals.tradier import TradierClient, TradierError, _as_list


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload, self.headers, self.calls = payload, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(self.payload)


def test_as_list():
    assert _as_list(None) == [] and _as_list({"a": 1}) == [{"a": 1}] and _as_list([1]) == [1]


def test_timesales_parsing():
    session = FakeSession({"series": {"data": [
        {"time": "2026-03-02T09:30:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10, "vwap": 1.2},
        {"time": "2026-03-02T09:31:00", "open": 1.5, "high": 2, "low": 1, "close": 1.8, "volume": 5, "vwap": 1.3},
    ]}})
    client = TradierClient("tok", "https://sandbox.tradier.com/v1", session=session)
    df = client.timesales("SPY", "2026-03-02 09:30", "2026-03-02 09:32")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2 and df["close"].iloc[-1] == 1.8
    assert session.headers["Authorization"] == "Bearer tok"
    assert session.calls[0][1]["session_filter"] == "all"


def test_single_quote_is_wrapped():
    client = TradierClient("tok", "u", session=FakeSession({"quotes": {"quote": {"symbol": "SPY", "last": 1}}}))
    assert client.quotes(["SPY"]) == [{"symbol": "SPY", "last": 1}]


def test_missing_token():
    with pytest.raises(TradierError):
        TradierClient("", "u")


def test_choose_expiration():
    exps = ["2026-03-01", "2026-03-02", "2026-03-04"]
    assert choose_expiration(exps, date(2026, 3, 2)) == "2026-03-02"
    assert choose_expiration(exps, date(2026, 3, 2), min_dte=1) == "2026-03-04"


def test_pick_contract_by_delta_and_spread():
    chain = [
        {"symbol": "A", "option_type": "call", "strike": 500, "bid": 2.0, "ask": 2.1, "greeks": {"delta": 0.52}},
        {"symbol": "B", "option_type": "call", "strike": 502, "bid": 1.0, "ask": 1.05, "greeks": {"delta": 0.44}},
        {"symbol": "C", "option_type": "call", "strike": 503, "bid": 0.5, "ask": 0.9, "greeks": {"delta": 0.45}},
        {"symbol": "D", "option_type": "put", "strike": 500, "bid": 2.0, "ask": 2.1, "greeks": {"delta": -0.45}},
    ]
    assert pick_contract(chain, 1, 501)["symbol"] == "B"  # C tiene spread enorme
    assert pick_contract(chain, -1, 501)["symbol"] == "D"


def test_contracts_for_risk():
    # $8000 * 1% = $80 de riesgo; prima $1.20 con stop 30% = $36 por contrato -> 2 contratos
    assert contracts_for_risk(8000, 1.0, 1.20, 30) == 2
    assert contracts_for_risk(8000, 1.0, 5.00, 30) == 0


def test_bias():
    assert bias_score(0.6, 510, 505, 500) == 2
    assert bias_score(-0.6, 499, 505, 500) == -2
    assert "ALCISTA" in bias_label(2) and "BAJISTA" in bias_label(-3) and "NEUTRAL" in bias_label(0)


def test_live_signal_on_last_closed_bar_and_alert():
    from market_signals.config import Settings
    from market_signals.monitor import format_alert, live_signal
    from tests.test_backtest import flat, make_day

    # la ruptura ocurre justo en la última barra cerrada
    ctx = make_day(flat(5) + [(100, 100.3, 99.95, 100.25)], prev_close=99.5)
    sig = live_signal(ctx, "orb", {"range_minutes": 5})
    assert sig is not None and sig.pos == len(ctx.c) - 1 and sig.direction == 1
    settings = Settings("t", "sandbox", "", "", "", 8000, 1.0, 30)
    contract = {"symbol": "SPY260302C00100000", "strike": 100, "bid": 1.1, "ask": 1.2,
                "spread_pct": 0.087, "greeks": {"delta": 0.45}}
    text = format_alert("SPY", "orb", ctx, sig, contract, settings)
    assert "CALL" in text and "2 contrato(s)" in text
