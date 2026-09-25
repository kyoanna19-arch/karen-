"""Cliente mínimo de la API de mercado de Tradier (solo lectura, no envía órdenes)."""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import requests


def _as_list(value: Any) -> list:
    """Tradier devuelve un dict cuando hay un solo elemento y una lista cuando hay varios."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


class TradierError(RuntimeError):
    pass


class TradierClient:
    def __init__(self, token: str, base_url: str, session: requests.Session | None = None, timeout: float = 15):
        if not token:
            raise TradierError("Falta TRADIER_TOKEN. Configúralo en tu archivo .env")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

    def _get(self, path: str, **params: Any) -> dict:
        resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        if resp.status_code != 200:
            raise TradierError(f"Tradier {path} respondió {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def clock(self) -> dict:
        return self._get("/markets/clock").get("clock", {})

    def quotes(self, symbols: Iterable[str], greeks: bool = False) -> list[dict]:
        data = self._get("/markets/quotes", symbols=",".join(symbols), greeks=str(greeks).lower())
        return _as_list((data.get("quotes") or {}).get("quote"))

    def daily_history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        data = self._get("/markets/history", symbol=symbol, interval="daily", start=start, end=end)
        rows = _as_list((data.get("history") or {}).get("day"))
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df.pop("date"))
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def timesales(self, symbol: str, start: str, end: str, interval: str = "1min",
                  session_filter: str = "all") -> pd.DataFrame:
        """Barras intradía. Horas en tiempo del Este (ET). start/end: 'YYYY-MM-DD HH:MM'."""
        data = self._get("/markets/timesales", symbol=symbol, interval=interval,
                         start=start, end=end, session_filter=session_filter)
        rows = _as_list((data.get("series") or {}).get("data"))
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df.pop("time"))
        df.index.name = "datetime"
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def option_expirations(self, symbol: str) -> list[str]:
        data = self._get("/markets/options/expirations", symbol=symbol, includeAllRoots="true")
        return _as_list((data.get("expirations") or {}).get("date"))

    def option_chain(self, symbol: str, expiration: str, greeks: bool = True) -> list[dict]:
        data = self._get("/markets/options/chains", symbol=symbol, expiration=expiration,
                         greeks=str(greeks).lower())
        return _as_list((data.get("options") or {}).get("option"))
