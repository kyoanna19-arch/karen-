"""Monitor en vivo: revisa cada minuto si la estrategia elegida da señal y te avisa por Telegram.

No envía órdenes. Tú decides si entras: el sistema te da el contrato sugerido, el tamaño y
los niveles de stop/objetivo.
"""
from __future__ import annotations

import time as systime
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .config import Settings
from .options import choose_expiration, contracts_for_risk, pick_contract
from .strategies import STRATEGIES, DayContext, Signal
from .tradier import TradierClient, TradierError

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET).replace(tzinfo=None)


def today_context(client: TradierClient, symbol: str, now: datetime) -> DayContext | None:
    today = now.date()
    bars = client.timesales(symbol, f"{today} 04:00", now.strftime("%Y-%m-%d %H:%M"), "1min", "all")
    bars = bars[bars.index + timedelta(minutes=1) <= now]  # descarta la barra que aún no cierra
    regular = bars.between_time("09:30", "15:59")
    if len(regular) < 2:
        return None
    hist = client.daily_history(symbol, str(today - timedelta(days=10)), str(today - timedelta(days=1)))
    hist = hist[hist.index.date < today]
    if hist.empty:
        return None
    prev = hist.iloc[-1]
    return DayContext(day=today, bars=regular, premarket=bars.between_time("04:00", "09:29"),
                      prev_close=float(prev["close"]), prev_high=float(prev["high"]), prev_low=float(prev["low"]))


def live_signal(ctx: DayContext, strategy: str, params: dict) -> Signal | None:
    """En vivo la señal puede estar en la última barra cerrada (en backtest se entra en la siguiente)."""
    ctx.live = True
    return STRATEGIES[strategy].fn(ctx, **params)


def format_alert(symbol: str, strategy: str, ctx: DayContext, sig: Signal, contract: dict | None,
                 settings: Settings) -> str:
    side = "CALL 🟢" if sig.direction == 1 else "PUT 🔴"
    price = float(ctx.c[sig.pos])
    risk = abs(price - sig.stop)
    target = sig.target_price if sig.target_price is not None else price + sig.direction * (sig.target_r or 1) * risk
    lines = [
        f"🚨 SEÑAL {symbol} — {side}",
        f"Estrategia: {STRATEGIES[strategy].description} ({strategy})",
        f"Motivo: {sig.note}",
        f"Hora señal: {ctx.times[sig.pos]:%H:%M} ET",
        f"Subyacente: {price:.2f} | Stop {sig.stop:.2f} | Objetivo {target:.2f}",
        f"Salida por tiempo: {sig.exit_by:%H:%M} ET",
    ]
    if contract:
        n = contracts_for_risk(settings.account_size, settings.risk_per_trade_pct, contract["ask"],
                               settings.option_stop_pct)
        delta = (contract.get("greeks") or {}).get("delta")
        lines += [
            "",
            f"Contrato: {contract['symbol']}",
            f"Strike {contract['strike']} | Bid {contract['bid']} / Ask {contract['ask']}"
            f" | Spread {contract['spread_pct']:.0%}" + (f" | Delta {delta:.2f}" if delta is not None else ""),
            f"Tamaño sugerido: {n} contrato(s) (riesgo {settings.risk_per_trade_pct}% de "
            f"${settings.account_size:,.0f}, stop opción -{settings.option_stop_pct:.0f}%)",
        ]
        if n == 0:
            lines.append("⚠️ La prima es muy cara para tu riesgo por trade. Mejor no entrar.")
    else:
        lines.append("\n⚠️ No encontré un contrato con spread aceptable.")
    lines.append("\n⚖️ Cuenta < $25k en margen: revisa cuántos day trades llevas (máx. 3 en 5 días hábiles).")
    return "\n".join(lines)


def find_contract(client: TradierClient, symbol: str, direction: int, price: float, today: date) -> dict | None:
    expiration = choose_expiration(client.option_expirations(symbol), today)
    if not expiration:
        return None
    return pick_contract(client.option_chain(symbol, expiration), direction, price)


def run_monitor(client: TradierClient, settings: Settings, notify, symbol: str, strategy: str,
                params: dict, until: str = "11:30", poll_seconds: int = 30) -> None:
    end = datetime.strptime(until, "%H:%M").time()
    notify(f"👀 Monitor activo: {symbol} con {strategy} {params or ''} hasta las {until} ET")
    while now_et().time() < end:
        now = now_et()
        try:
            ctx = today_context(client, symbol, now)
            sig = live_signal(ctx, strategy, params) if ctx else None
            if sig is not None and sig.pos < len(ctx.c) - 3:
                notify(f"Monitor {symbol}: la señal de {strategy} ya ocurrió a las "
                       f"{ctx.times[sig.pos]:%H:%M} ET. Es tarde para entrar; no persigas el precio.")
                return
            if sig is not None:
                contract = find_contract(client, symbol, sig.direction, float(ctx.c[sig.pos]), now.date())
                notify(format_alert(symbol, strategy, ctx, sig, contract, settings))
                return  # una señal por día
        except TradierError as exc:
            print(f"[{now:%H:%M:%S}] error de Tradier: {exc}")
        systime.sleep(poll_seconds)
    notify(f"Monitor {symbol}: sin señal de {strategy} hoy. No operar también es una decisión.")
