"""Datos FALSOS (caminata aleatoria) para probar el sistema sin conexión.
Los resultados del backtest con estos datos NO significan nada: solo sirven para ver que todo corre."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_bars(days: int = 120, start: str = "2026-01-05", price: float = 500.0,
                   seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for day in pd.bdate_range(start, periods=days):
        price *= 1 + rng.normal(0, 0.006)  # gap nocturno
        idx = pd.date_range(day + pd.Timedelta(hours=4), day + pd.Timedelta(hours=15, minutes=59), freq="1min")
        regular = idx.time >= pd.Timestamp("09:30").time()
        vol = np.where(regular, 0.0006, 0.0002)
        drift = rng.normal(0, 0.000005)
        closes = price * np.cumprod(1 + drift + rng.normal(0, 1, len(idx)) * vol)
        opens = np.concatenate([[price], closes[:-1]])
        wiggle = np.abs(rng.normal(0, 1, len(idx))) * vol * closes
        frames.append(pd.DataFrame({
            "open": opens, "high": np.maximum(opens, closes) + wiggle, "low": np.minimum(opens, closes) - wiggle,
            "close": closes, "volume": np.where(regular, rng.integers(50_000, 300_000, len(idx)),
                                                rng.integers(1_000, 10_000, len(idx))).astype(float),
        }, index=idx))
        price = closes[-1]
    df = pd.concat(frames)
    df.index.name = "datetime"
    return df
