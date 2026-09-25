from datetime import date

import pytest

from market_signals.backtest import build_contexts, load_bars_csv
from market_signals.polygon import PolygonClient, PolygonError, download_history, month_ranges, results_to_frame


def bar(ts_utc, price, vol=1000):
    import pandas as pd
    ms = int(pd.Timestamp(ts_utc, tz="UTC").timestamp() * 1000)
    return {"t": ms, "o": price, "h": price + 0.1, "l": price - 0.1, "c": price, "v": vol}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.content, self.text = b"x", str(payload)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


def test_month_ranges():
    assert month_ranges(date(2024, 1, 15), date(2024, 3, 10)) == [
        (date(2024, 1, 15), date(2024, 1, 31)), (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 10))]


def test_timestamps_converted_to_eastern_time():
    # 14:30 UTC en marzo (antes del cambio de horario) = 9:30 ET; en julio 13:30 UTC = 9:30 ET
    df = results_to_frame([bar("2024-03-04 14:30", 500), bar("2024-07-01 13:30", 540),
                           bar("2024-07-01 01:00", 1)])  # 21:00 ET: fuera de horario, se descarta
    assert [t.strftime("%Y-%m-%d %H:%M") for t in df.index] == ["2024-03-04 09:30", "2024-07-01 09:30"]


def test_pagination_and_rate_limit_retry():
    session = FakeSession([
        FakeResponse({}, status=429),
        FakeResponse({"status": "OK", "results": [bar("2024-03-04 14:30", 500)],
                      "next_url": "https://api.polygon.io/next"}),
        FakeResponse({"status": "OK", "results": [bar("2024-03-04 14:31", 501)]}),
    ])
    waits = []
    client = PolygonClient("key", session=session, sleep=waits.append)
    df = client.minute_bars("spy", date(2024, 3, 1), date(2024, 3, 31))
    assert len(df) == 2 and 60 in waits
    assert "/SPY/range/1/minute/2024-03-01/2024-03-31" in session.calls[0][0]
    assert session.calls[2] == ("https://api.polygon.io/next", {"apiKey": "key"})


def test_not_authorized_raises():
    session = FakeSession([FakeResponse({"status": "NOT_AUTHORIZED", "message": "plan"}, status=403)])
    with pytest.raises(PolygonError, match="plan"):
        PolygonClient("key", session=session, sleep=lambda s: None).minute_bars("SPY", date(2020, 1, 1),
                                                                                 date(2020, 1, 31))


def test_missing_key():
    with pytest.raises(PolygonError):
        PolygonClient("")


def test_download_skips_unauthorized_month_and_builds_backtest_file(tmp_path):
    import pandas as pd
    days = pd.bdate_range("2024-02-05", "2024-02-09")
    results = []
    for d in days:  # 4:00 a 16:00 ET = 9:00 a 21:00 UTC en febrero
        for m in range(0, 12 * 60):
            results.append(bar(d + pd.Timedelta(hours=9, minutes=m), 500 + m * 0.001))
    session = FakeSession([
        FakeResponse({"status": "NOT_AUTHORIZED", "message": "fuera de tu plan"}, status=403),
        FakeResponse({"status": "OK", "results": results}),
    ])
    client = PolygonClient("key", session=session, sleep=lambda s: None)
    path = download_history(client, "SPY", date(2024, 1, 1), date(2024, 2, 29), folder=tmp_path)
    bars = load_bars_csv(path)
    assert bars.index.min().strftime("%H:%M") == "04:00"
    contexts = build_contexts(bars)
    assert len(contexts) == 4 and len(contexts[0].premarket) == 330 and len(contexts[0].bars) == 390

    # correrlo otra vez no duplica
    session.responses = [FakeResponse({"status": "OK", "results": results})]
    download_history(client, "SPY", date(2024, 2, 1), date(2024, 2, 29), folder=tmp_path)
    assert len(load_bars_csv(path)) == len(bars)
