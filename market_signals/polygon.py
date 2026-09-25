"""Descarga de historial de velas de 1 minuto desde Polygon.io (incluye pre-market).

Plan gratuito: ~2 años de historia y 5 consultas por minuto. Pedimos un mes por consulta,
así que 2 años tardan unos 5 minutos. Cada mes se guarda al llegar: si se corta,
vuelves a correr el comando y continúa sin duplicar datos.
"""
from __future__ import annotations

import time as systime
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from .data import merge_into_cache

BASE_URL = "https://api.polygon.io"


class PolygonError(RuntimeError):
    pass


def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """[(1-ene, 31-ene), (1-feb, 29-feb), ...] recortado a start/end."""
    ranges = []
    for month_start in pd.date_range(start.replace(day=1), end, freq="MS"):
        month_end = (month_start + pd.offsets.MonthEnd(0)).date()
        ranges.append((max(start, month_start.date()), min(end, month_end)))
    return ranges


def results_to_frame(results: list[dict]) -> pd.DataFrame:
    """Convierte la respuesta de Polygon (t = milisegundos UTC) a velas en hora del Este."""
    if not results:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(results)
    index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    out = pd.DataFrame({"open": df["o"].to_numpy(float), "high": df["h"].to_numpy(float),
                        "low": df["l"].to_numpy(float), "close": df["c"].to_numpy(float),
                        "volume": df["v"].to_numpy(float)}, index=pd.DatetimeIndex(index, name="datetime"))
    return out.between_time("04:00", "19:59")


class PolygonClient:
    def __init__(self, api_key: str, session: requests.Session | None = None, base_url: str = BASE_URL,
                 pause_seconds: float = 12.5, sleep: Callable[[float], None] = systime.sleep):
        if not api_key:
            raise PolygonError("Falta POLYGON_API_KEY en tu archivo .env")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.pause_seconds = pause_seconds  # plan gratuito: 5 consultas por minuto
        self.sleep = sleep

    def _get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(5):
            resp = self.session.get(url, params={**(params or {}), "apiKey": self.api_key}, timeout=30)
            if resp.status_code == 429:  # demasiadas consultas: esperar y reintentar
                print("  Límite de consultas alcanzado, esperando 60 s...")
                self.sleep(60)
                continue
            data = resp.json() if resp.content else {}
            if resp.status_code != 200 or data.get("status") in ("ERROR", "NOT_AUTHORIZED"):
                msg = data.get("message") or data.get("error") or resp.text[:300]
                raise PolygonError(f"Polygon respondió {resp.status_code}: {msg}")
            return data
        raise PolygonError("Polygon siguió rechazando por límite de consultas")

    def minute_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        url = f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/range/1/minute/{start}/{end}"
        params: dict | None = {"adjusted": "true", "sort": "asc", "limit": 50000}
        frames = []
        while url:
            data = self._get(url, params)
            frames.append(results_to_frame(data.get("results") or []))
            url, params = data.get("next_url"), None  # next_url ya trae los parámetros
            self.sleep(self.pause_seconds)
        return pd.concat(frames) if frames else results_to_frame([])


def download_history(client: PolygonClient, symbol: str, start: date, end: date,
                     folder: str | Path = "data") -> Path | None:
    path = None
    for month_start, month_end in month_ranges(start, end):
        try:
            bars = client.minute_bars(symbol, month_start, month_end)
        except PolygonError as exc:
            # el plan gratuito no cubre fechas muy viejas: se salta ese mes y sigue
            print(f"  {month_start:%Y-%m}: {exc}")
            continue
        if bars.empty:
            print(f"  {month_start:%Y-%m}: sin datos")
            continue
        path = merge_into_cache([bars], symbol, folder)
        days = len(set(bars.index.date))
        print(f"  {month_start:%Y-%m}: {len(bars):,} velas, {days} días")
    return path
