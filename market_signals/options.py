"""Selección del contrato de opción y tamaño de la posición según tu riesgo."""
from __future__ import annotations

import math
from datetime import date


def choose_expiration(expirations: list[str], today: date, min_dte: int = 0) -> str | None:
    """La expiración más cercana con al menos `min_dte` días (0 = 0DTE si existe hoy)."""
    valid = sorted(e for e in expirations if (date.fromisoformat(e) - today).days >= min_dte)
    return valid[0] if valid else None


def pick_contract(chain: list[dict], direction: int, underlying_price: float, target_delta: float = 0.45,
                  max_spread_pct: float = 0.10) -> dict | None:
    """CALL si direction=1, PUT si direction=-1. Busca el delta más cercano a `target_delta`
    con un spread bid/ask aceptable (en scalping el spread es el costo más grande)."""
    option_type = "call" if direction == 1 else "put"
    candidates = []
    for opt in chain:
        if opt.get("option_type") != option_type:
            continue
        bid, ask = opt.get("bid") or 0, opt.get("ask") or 0
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid
        if spread_pct > max_spread_pct:
            continue
        delta = (opt.get("greeks") or {}).get("delta")
        if delta is not None:
            distance = abs(abs(delta) - target_delta)
        else:  # sin griegas: el strike más cercano al precio
            distance = abs(opt["strike"] - underlying_price) / underlying_price
        candidates.append((distance, spread_pct, {**opt, "mid": round(mid, 2), "spread_pct": round(spread_pct, 3)}))
    if not candidates:
        return None
    return min(candidates, key=lambda c: (c[0], c[1]))[2]


def contracts_for_risk(account_size: float, risk_pct: float, premium: float, stop_pct: float) -> int:
    """Número de contratos para que, si la opción pierde `stop_pct`% de la prima,
    la pérdida no pase de `risk_pct`% de la cuenta."""
    risk_dollars = account_size * risk_pct / 100
    loss_per_contract = premium * 100 * stop_pct / 100
    if loss_per_contract <= 0:
        return 0
    return math.floor(risk_dollars / loss_per_contract)
