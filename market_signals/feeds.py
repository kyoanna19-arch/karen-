"""Feed de datos de Tradier para paper.py (tiempo real gratis si tienes cuenta en Tradier)."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from .options import choose_expiration
from .tradier import TradierClient, TradierError


class TradierFeed:
    def __init__(self, client: TradierClient):
        self.client = client

    def minute_bars(self, symbol: str, now: datetime) -> pd.DataFrame:
        return self.client.timesales(symbol, f"{now:%Y-%m-%d} 04:00", f"{now:%Y-%m-%d %H:%M}", "1min", "all")

    def prev_day(self, symbol: str, today: date) -> dict:
        hist = self.client.daily_history(symbol, str(today - timedelta(days=10)), str(today - timedelta(days=1)))
        hist = hist[hist.index.date < today]
        if hist.empty:
            raise TradierError(f"No hay datos del día anterior para {symbol}")
        row = hist.iloc[-1]
        return {"close": float(row["close"]), "high": float(row["high"]), "low": float(row["low"])}

    def option_candidates(self, symbol: str, direction: int, price: float, today: date) -> list[dict]:
        exp = choose_expiration(self.client.option_expirations(symbol), today)
        if exp is None:
            return []
        chain = self.client.option_chain(symbol, exp)
        return [{**opt, "expiration": exp, "underlying": opt.get("underlying") or symbol} for opt in chain]
