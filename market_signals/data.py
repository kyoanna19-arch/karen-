"""Descarga y guarda barras de 1 minuto desde Tradier para ir armando tu propio historial.

Tradier solo guarda unas semanas de barras de 1 minuto, así que conviene correr la
descarga cada semana: el archivo local va creciendo y el backtest se vuelve más confiable.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .backtest import load_bars_csv
from .tradier import TradierClient, TradierError


def cache_path(symbol: str, folder: str | Path = "data") -> Path:
    return Path(folder) / f"{symbol.upper()}_1min.csv"


def download_recent(client: TradierClient, symbol: str, days: int = 20, folder: str | Path = "data",
                    today: date | None = None) -> Path:
    today = today or date.today()
    frames = []
    d = today - timedelta(days=int(days * 1.5) + 3)  # días calendario de sobra para cubrir fines de semana
    while d <= today:
        if d.weekday() < 5:
            try:
                bars = client.timesales(symbol, f"{d} 04:00", f"{d} 20:00", "1min", "all")
            except TradierError as exc:
                print(f"  {d}: {exc}")
                bars = pd.DataFrame()
            if not bars.empty:
                frames.append(bars)
                print(f"  {d}: {len(bars)} barras")
        d += timedelta(days=1)

    if not frames:
        raise TradierError(f"No llegaron datos para {symbol}")
    return merge_into_cache(frames, symbol, folder)


def merge_into_cache(frames: list[pd.DataFrame], symbol: str, folder: str | Path = "data") -> Path:
    """Junta barras nuevas con el archivo guardado (sin duplicados) y lo reescribe ordenado."""
    path = cache_path(symbol, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [f for f in frames if not f.empty]
    if path.exists():
        frames.insert(0, load_bars_csv(path))
    if not frames:
        raise ValueError(f"No hay barras para guardar de {symbol}")
    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_csv(path, index_label="datetime")
    return path
