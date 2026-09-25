"""Reporte antes de la apertura: contexto del mercado, sesgo del día y niveles clave."""
from __future__ import annotations

from datetime import date, timedelta

from .tradier import TradierClient, TradierError

INDEX_SYMBOLS = ("SPY", "QQQ", "IWM")
WATCHLIST = ("AAPL", "NVDA", "TSLA", "AMD", "META", "AMZN", "MSFT", "GOOGL")
GAP_THRESHOLD = 0.25  # % de gap que cuenta como movimiento


def _pct(a: float, b: float) -> float:
    return (a - b) / b * 100 if b else 0.0


def bias_score(gap_pct: float, last: float, prev_high: float, prev_low: float) -> int:
    """+1/-1 por el gap y +1/-1 si el pre-market está fuera del rango de ayer."""
    score = 0
    if gap_pct > GAP_THRESHOLD:
        score += 1
    elif gap_pct < -GAP_THRESHOLD:
        score -= 1
    if last > prev_high:
        score += 1
    elif last < prev_low:
        score -= 1
    return score


def bias_label(total: int) -> str:
    if total >= 2:
        return "ALCISTA 🟢 (prioriza CALLS; ORB/pre-market break a favor del gap)"
    if total <= -2:
        return "BAJISTA 🔴 (prioriza PUTS; ORB/pre-market break a favor del gap)"
    return "NEUTRAL / RANGO ⚪ (cuidado con rupturas falsas; considera gap fade o esperar)"


def build_report(client: TradierClient, today: date | None = None, index_symbols=INDEX_SYMBOLS,
                 watchlist=WATCHLIST) -> str:
    today = today or date.today()
    quotes = {q["symbol"]: q for q in client.quotes([*index_symbols, *watchlist, "VIX"])}
    lines = [f"📊 REPORTE PRE-MARKET {today:%Y-%m-%d}", ""]
    total = 0

    for sym in index_symbols:
        q = quotes.get(sym)
        if not q:
            continue
        hist = client.daily_history(sym, str(today - timedelta(days=10)), str(today - timedelta(days=1)))
        hist = hist[hist.index.date < today]
        if hist.empty:
            continue
        prev = hist.iloc[-1]
        pm = client.timesales(sym, f"{today} 04:00", f"{today} 09:30", "1min", "all")
        last = float(q.get("last") or prev["close"])
        gap = _pct(last, prev["close"])
        score = bias_score(gap, last, prev["high"], prev["low"])
        if sym in ("SPY", "QQQ"):
            total += score
        lines.append(f"{sym}: {last:.2f} ({gap:+.2f}% vs cierre {prev['close']:.2f})")
        lines.append(f"   Ayer  H {prev['high']:.2f} / L {prev['low']:.2f}")
        if not pm.empty:
            lines.append(f"   Pre-M H {pm['high'].max():.2f} / L {pm['low'].min():.2f}"
                         f"  vol {int(pm['volume'].sum()):,}")

    vix = quotes.get("VIX")
    if vix and vix.get("last"):
        change = vix.get("change_percentage") or 0
        lines.append(f"VIX: {vix['last']:.2f} ({change:+.2f}%)")
        if change > 5:
            total -= 1
        elif change < -5:
            total += 1
        if vix["last"] >= 25:
            lines.append("   ⚠️ VIX alto: primas caras y movimientos bruscos, reduce tamaño")

    lines += ["", f"SESGO: {bias_label(total)}", "", "Movimientos en la watchlist:"]
    movers = []
    for sym in watchlist:
        q = quotes.get(sym)
        if q and q.get("last") and q.get("prevclose"):
            movers.append((_pct(q["last"], q["prevclose"]), sym, q["last"]))
    for gap, sym, last in sorted(movers, key=lambda m: -abs(m[0]))[:5]:
        lines.append(f"   {sym:6s} {last:>9.2f}  {gap:+.2f}%")

    lines += ["", "Recuerda: revisa el calendario económico (CPI, FOMC, empleo) antes de operar."]
    return "\n".join(lines)


def safe_report(client: TradierClient, **kwargs) -> str:
    try:
        return build_report(client, **kwargs)
    except TradierError as exc:
        return f"No se pudo generar el reporte: {exc}"
