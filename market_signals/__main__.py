"""Uso: python -m market_signals <comando> [opciones]

Comandos:
  premarket   Reporte antes de la apertura (sesgo del día y niveles)
  monitor     Vigila una estrategia en vivo y alerta la señal
  download    Descarga barras de 1 minuto de Tradier a data/
  history     Descarga años de historial de 1 minuto desde Polygon.io
  options-study  Estudia el stop sobre la prima y las salidas escalonadas
  ibkr-test   Revisa la conexión con tu cuenta paper de Interactive Brokers
  paper       Opera la estrategia automáticamente en tu cuenta PAPER de IBKR
  paper-journal  Resumen de los trades hechos en paper
  backtest    Compara estrategias con datos históricos (mes por mes)
  telegram-test  Envía un mensaje de prueba a tu Telegram
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import backtest as bt
from .config import load_settings
from .strategies import STRATEGIES
from .telegram import Notifier


def _value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_params(text: str | None) -> dict:
    """Acepta JSON ('{"min_score":9}') o la forma simple 'min_score=9,timeframe=5'.
    La forma simple funciona en PowerShell de Windows, que a veces borra las comillas del JSON."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise SystemExit("No entendí --params. En Windows usa la forma simple, ej. --params min_score=9")
    params = {}
    for part in text.split(","):
        if "=" not in part:
            raise SystemExit(f"Parámetro sin '=': {part}. Ejemplo: --params min_score=9,target_r=1.5")
        key, value = part.split("=", 1)
        params[key.strip()] = _value(value.strip())
    return params


def _client(settings):
    from .tradier import TradierClient
    return TradierClient(settings.tradier_token, settings.tradier_base_url)


def _load_contexts(args) -> list:
    if args.demo:
        from .synthetic import synthetic_bars
        print("⚠️ DATOS SINTÉTICOS: los números no significan nada, solo prueban que el sistema corre.\n")
        bars = synthetic_bars(days=args.demo_days)
    else:
        bars = bt.load_bars_csv(args.data)
    contexts = bt.build_contexts(bars)
    if contexts:
        print(f"Días analizados: {len(contexts)} ({contexts[0].day} a {contexts[-1].day})\n")
    return contexts


def cmd_backtest(args) -> None:
    contexts = _load_contexts(args)
    if len(contexts) < 20:
        print("Muy pocos días para sacar conclusiones (mínimo 20, ideal 250+).")
        return
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.strategy:
        params = parse_params(args.params)
        trades = bt.run_strategy(contexts, args.strategy, params, args.cost_r)
        print(f"== {args.strategy} {params} ==")
        for k, v in bt.summarize(trades).items():
            print(f"  {k:22s} {v}")
        print("\nResultados por mes (R neto):")
        print(bt.monthly_table(trades).to_string())
        for title, table in bt.breakdown(trades).items():
            print(f"\n{title}:\n{table.to_string()}")
        impact = bt.condition_impact(trades)
        if not impact.empty:
            print("\n¿Qué condiciones del score importan? (usa min_score bajo para ver más casos):")
            print(impact.to_string(index=False))
        if STRATEGIES[args.strategy].ablations:
            print("\n¿Qué regla aporta? (cada fila apaga una regla):")
            print(bt.ablation(contexts, args.strategy, params, args.cost_r).to_string())
        bt.trades_frame(trades).to_csv(out / f"trades_{args.strategy}.csv", index=False)
        print(f"\nTrades guardados en {out / f'trades_{args.strategy}.csv'}")
        return

    print("== Ranking de todas las combinaciones (todo el periodo) ==")
    names = args.only.split(",") if args.only else None
    ranked = bt.optimize(contexts, names, cost_r=args.cost_r, min_trades=args.min_trades)
    if ranked.empty:
        print("Ninguna combinación tuvo suficientes trades.")
        return
    cols = ["strategy", "params", "trades", "win_rate", "avg_r", "total_r", "profit_factor",
            "max_drawdown_r", "pct_months_positive", "worst_month_r"]
    print(ranked[cols].head(args.top).to_string())
    ranked.to_csv(out / "ranking.csv", index=False)

    print("\n== Walk-forward: optimiza con el 70% inicial, prueba en el 30% final ==")
    wf = bt.walk_forward(contexts, cost_r=args.cost_r, min_trades=args.min_trades, names=names)
    print(wf.to_string() if not wf.empty else "Sin resultados.")
    wf.to_csv(out / "walk_forward.csv", index=False)
    print("\nCómo leerlo: una estrategia vale la pena si test_avg_r > 0 con suficientes trades y "
          "la mayoría de los meses son positivos. Si solo brilla en entrenamiento, es sobreajuste.")
    print(f"Archivos guardados en {out}/")


def cmd_premarket(args) -> None:
    from .premarket import safe_report
    settings = load_settings()
    Notifier(settings.telegram_bot_token, settings.telegram_chat_id, args.send)(safe_report(_client(settings)))


def cmd_monitor(args) -> None:
    from .monitor import run_monitor
    settings = load_settings()
    notify = Notifier(settings.telegram_bot_token, settings.telegram_chat_id, args.send)
    params = parse_params(args.params)
    run_monitor(_client(settings), settings, notify, args.symbol.upper(), args.strategy, params,
                until=args.until, poll_seconds=args.poll)


def cmd_download(args) -> None:
    from .data import download_recent
    settings = load_settings()
    path = download_recent(_client(settings), args.symbol, days=args.days)
    print(f"Guardado en {path}")


def cmd_history(args) -> None:
    import os
    from datetime import date, timedelta
    from .config import load_dotenv
    from .polygon import PolygonClient, download_history
    load_dotenv()
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=int(args.years * 365))
    client = PolygonClient(os.environ.get("POLYGON_API_KEY", ""), pause_seconds=args.pause)
    for symbol in args.symbol.upper().split(","):
        print(f"Descargando {symbol} de {start} a {end} (el plan gratuito tarda ~15 s por mes)...")
        path = download_history(client, symbol, start, end)
        print(f"Listo: {path}" if path else f"No se descargó nada de {symbol}.")


def cmd_options_study(args) -> None:
    from . import options_sim as osim
    contexts = _load_contexts(args)
    params = parse_params(args.params)
    pairs = osim.signals_for(contexts, STRATEGIES[args.strategy].fn, params)
    pd.set_option("display.width", 220)
    print(f"Estrategia {args.strategy} {params}: {len(pairs)} señales en {len(contexts)} días")
    if len(pairs) < 10:
        print("Muy pocas señales para estudiar stops y salidas.")
        return

    def run(plan, iv=args.iv, delta=args.delta, dte=args.dte):
        trades = [t for t in (osim.simulate_option(c, s, args.strategy, plan, iv=iv, delta=delta, dte=dte,
                                                   slip=args.slip) for c, s in pairs) if t]
        st = bt.summarize(trades)
        avg_pct = sum(t.r for t in trades) * plan.stop_pct / len(trades) if trades else 0
        return trades, {"trades": st["trades"], "win_rate": st["win_rate"], "avg_%_prima": round(avg_pct * 100, 1),
                        "avg_r": st["avg_r"], f"$_por_trade_riesgo_{args.risk:.0f}": round(st["avg_r"] * args.risk, 2),
                        "total_r": st["total_r"], "max_dd_r": st["max_drawdown_r"],
                        "meses_+": st["pct_months_positive"], "peor_mes_r": st["worst_month_r"]}

    study = osim.stop_study(pairs, iv=args.iv, slip=args.slip)
    print(f"\nPrima promedio estimada: ${study['avg_premium']} (IV {args.iv:.0%}, opción {args.dte}DTE "
          f"delta ~{args.delta}) — las secciones 1 a 5 usan esta opción")
    print("\n== 1. ¿Cuánto cae la opción antes de ganar? ==")
    print(f"Peor caída de todos los trades (percentiles): {study['mae_percentiles']}")
    print("De los trades que llegaron a cada meta, % que un stop habría sacado ANTES:")
    print(pd.DataFrame(study["table"]).to_string(index=False))

    print("\n== 2. Comparación de stops (misma salida: mitad en +25%, mitad en +50%) ==")
    print("Con riesgo fijo en dólares, compara avg_r: un stop más corto permite más contratos.")
    rows = []
    for stop in osim.STOP_LEVELS:
        _, row = run(osim.ExitPlan(stop_pct=stop, tiers=((0.25, 0.5), (0.5, 0.5)), breakeven=False))
        rows.append({"stop": f"-{int(stop * 100)}%", **row})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n== 3. Salidas escalonadas con stop de -{int(args.stop * 100)}% ==")
    rows, best = [], None
    for plan in osim.EXIT_PLANS:
        plan = osim.ExitPlan(stop_pct=args.stop, tiers=plan.tiers, breakeven=plan.breakeven)
        trades, row = run(plan)
        rows.append({"plan": plan.label(), **row})
        if best is None or row["avg_r"] > best[2]["avg_r"]:
            best = (plan, trades, row)
    print(pd.DataFrame(rows).to_string(index=False))

    plan, trades, _ = best
    print(f"\n== 4. ¿Depende de la volatilidad? Mejor plan: {plan.label()} ==")
    rows = [{"IV": f"{iv:.0%}", **run(plan, iv)[1]} for iv in (0.12, 0.18, 0.25, 0.35)]
    print(pd.DataFrame(rows).to_string(index=False))
    print("Si el resultado cambia de signo según la IV, la ventaja es frágil.")

    print("\n== 6. ¿Qué opción conviene? (mejor plan, stop igual) ==")
    print("DTE = días al vencimiento (0 = vence hoy). Delta 0.45 ≈ ATM; 0.60-0.70 = dentro del dinero (ITM).")
    print("Más DTE y más delta = menos pérdida por tiempo (theta), pero prima más cara.")
    rows = []
    for dte in (0, 1, 2, 5):
        for delta in (0.45, 0.60, 0.70):
            rows.append({"DTE": dte, "delta": delta, **run(plan, delta=delta, dte=dte)[1]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    best_opt = table.sort_values("avg_r", ascending=False).iloc[0]
    best_dte, best_delta = int(best_opt["DTE"]), float(best_opt["delta"])

    print(f"\n== 7. Salida rápida: si no funcionó en N minutos, fuera ({best_dte}DTE delta {best_delta}) ==")
    rows = []
    for minutes in (10, 20, 30, 45, None):
        timed = osim.ExitPlan(stop_pct=plan.stop_pct, tiers=plan.tiers, breakeven=plan.breakeven, max_minutes=minutes)
        rows.append({"máx_minutos": minutes or "sin límite", **run(timed, delta=best_delta, dte=best_dte)[1]})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n== 5. Mes por mes con el mejor plan ({args.dte}DTE delta {args.delta}) ==")
    print(bt.monthly_table(trades).to_string())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bt.trades_frame(trades).to_csv(out / f"option_trades_{args.strategy}.csv", index=False)
    print(f"\nTrades guardados en {out / f'option_trades_{args.strategy}.csv'}")
    print("Recuerda: el precio de la opción es estimado con Black-Scholes. El paper trading confirma con precios reales.")


def _ibkr(client_id: int | None = None):
    import os
    from .ibkr import IBKRBroker
    return IBKRBroker(os.environ.get("IBKR_HOST", "127.0.0.1"), int(os.environ.get("IBKR_PORT", "7497")),
                      client_id or int(os.environ.get("IBKR_CLIENT_ID", "7")))


def cmd_ibkr_test(args) -> None:
    from datetime import date
    from .monitor import now_et
    settings = load_settings()
    broker = _ibkr()
    try:
        print("✅ Conectado a IBKR en modo PAPER")
        print(broker.summary())
        bars = broker.minute_bars(args.symbol, now_et())
        print(f"Velas de 1 minuto de hoy para {args.symbol}: {len(bars)}"
              + (f" (última {bars.index[-1]:%H:%M} ET, {bars['close'].iloc[-1]:.2f})" if len(bars) else ""))
        print(f"Día anterior: {broker.prev_day(args.symbol, date.today())}")
        price = float(bars["close"].iloc[-1]) if len(bars) else broker.prev_day(args.symbol, date.today())["close"]
        options = broker.option_candidates(args.symbol, 1, price, date.today())
        print(f"Calls cerca del precio: {len(options)}")
        for o in options[:4]:
            print(f"   {o['symbol']}  bid {o['bid']}  ask {o['ask']}  delta {(o['greeks'] or {}).get('delta')}")
        if options and all(o["bid"] == 0 for o in options):
            print("⚠️ No llegan precios de opciones: revisa tu suscripción de datos de mercado (OPRA) "
                  "o usa --feed tradier en el comando paper.")
    finally:
        broker.disconnect()
    if settings.tradier_token:
        print("Tradier también está configurado: puedes usar --feed tradier (recomendado).")


def parse_tiers(text: str) -> tuple[tuple[float, float], ...]:
    """'25:50,50:50' -> ((0.25, 0.5), (0.5, 0.5)): vender 50% en +25% y 50% en +50%."""
    tiers = []
    for part in text.split(","):
        gain, frac = part.split(":")
        tiers.append((float(gain) / 100, float(frac) / 100))
    if sum(f for _, f in tiers) > 1.0001:
        raise SystemExit("Los porcentajes de los escalones suman más de 100%")
    return tuple(sorted(tiers))


def cmd_paper(args) -> None:
    from .monitor import now_et
    from .paper import PaperTrader
    settings = load_settings()
    notify = Notifier(settings.telegram_bot_token, settings.telegram_chat_id, args.send)
    broker = _ibkr()
    try:
        notify(f"Conectado a IBKR PAPER. {broker.summary()}")
        if args.feed == "ibkr":
            feed = broker
        else:
            from .feeds import TradierFeed
            feed = TradierFeed(_client(settings))
        params = parse_params(args.params)
        plan = None
        if args.exit_mode == "premium":
            from .options_sim import ExitPlan
            plan = ExitPlan(stop_pct=args.stop_pct / 100, tiers=parse_tiers(args.tiers), breakeven=args.breakeven,
                            max_minutes=args.max_minutes or None)
            notify(f"Salida: stop -{args.stop_pct:.0f}% de la prima | {plan.label()}")
        PaperTrader(feed, broker, settings, notify, args.symbol.upper(), args.strategy, params,
                    now=now_et, sleep=broker.sleep, poll_seconds=args.poll, until=args.until,
                    exit_plan=plan, min_dte=args.dte, target_delta=args.delta).run()
    finally:
        broker.disconnect()


def cmd_paper_journal(args) -> None:
    from .paper import journal_summary
    print(journal_summary())


def cmd_telegram_test(args) -> None:
    settings = load_settings()
    if not settings.telegram_enabled:
        print("Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en .env")
        return
    Notifier(settings.telegram_bot_token, settings.telegram_chat_id, True)("✅ Conexión con Telegram funcionando.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="market_signals", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backtest", help="Comparar estrategias con datos históricos")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV de barras de 1 minuto (ej. data/SPY_1min.csv)")
    src.add_argument("--demo", action="store_true", help="Usar datos sintéticos de prueba")
    p.add_argument("--demo-days", type=int, default=120)
    p.add_argument("--strategy", choices=list(STRATEGIES), help="Analizar una sola estrategia en detalle")
    p.add_argument("--params", help='Parámetros JSON, ej. \'{"range_minutes": 15, "target_r": 1.5}\'')
    p.add_argument("--only", help="Comparar solo estas estrategias, ej. ema_vwap_rsi,orb")
    p.add_argument("--cost-r", type=float, default=0.1, help="Costo por trade en R (spread/slippage)")
    p.add_argument("--min-trades", type=int, default=15)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out", default="reports")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("premarket", help="Reporte pre-market")
    p.add_argument("--send", action="store_true", help="Enviar a Telegram")
    p.set_defaults(func=cmd_premarket)

    p = sub.add_parser("monitor", help="Vigilar una estrategia en vivo")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--strategy", choices=list(STRATEGIES), default="orb")
    p.add_argument("--params", help="Parámetros JSON (usa los que ganaron en el backtest)")
    p.add_argument("--until", default="11:30", help="Hora ET para dejar de vigilar")
    p.add_argument("--poll", type=int, default=30, help="Segundos entre revisiones")
    p.add_argument("--send", action="store_true", help="Enviar a Telegram")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("download", help="Descargar barras de 1 minuto")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--days", type=int, default=20)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("history", help="Descargar historial largo desde Polygon.io")
    p.add_argument("--symbol", default="SPY", help="Uno o varios separados por coma, ej. SPY,QQQ")
    p.add_argument("--years", type=float, default=2)
    p.add_argument("--start", help="Fecha inicial YYYY-MM-DD (opcional)")
    p.add_argument("--end", help="Fecha final YYYY-MM-DD (por defecto ayer)")
    p.add_argument("--pause", type=float, default=12.5,
                   help="Segundos entre consultas (12.5 para el plan gratuito, 0 si pagas)")
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("options-study", help="Estudiar stop sobre la prima y salidas escalonadas")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV de barras de 1 minuto")
    src.add_argument("--demo", action="store_true", help="Usar datos sintéticos de prueba")
    p.add_argument("--demo-days", type=int, default=250)
    p.add_argument("--strategy", choices=list(STRATEGIES), default="karen_score")
    p.add_argument("--params", help="Parámetros JSON de la estrategia")
    p.add_argument("--stop", type=float, default=0.25, help="Stop sobre la prima (0.25 = 25%%)")
    p.add_argument("--iv", type=float, default=0.18, help="Volatilidad implícita supuesta (0.18 = 18%%)")
    p.add_argument("--dte", type=int, default=0, help="Días al vencimiento de la opción (0 = vence hoy)")
    p.add_argument("--delta", type=float, default=0.45, help="Delta de la opción (0.45 ≈ ATM, 0.70 = ITM)")
    p.add_argument("--slip", type=float, default=0.01, help="Costo de spread por lado (0.01 = 1%%)")
    p.add_argument("--risk", type=float, default=80, help="Dólares en riesgo por trade")
    p.add_argument("--out", default="reports")
    p.set_defaults(func=cmd_options_study)

    p = sub.add_parser("ibkr-test", help="Probar la conexión con IBKR paper")
    p.add_argument("--symbol", default="SPY")
    p.set_defaults(func=cmd_ibkr_test)

    p = sub.add_parser("paper", help="Operar automáticamente en la cuenta PAPER de IBKR")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--strategy", choices=list(STRATEGIES), default="karen_score")
    p.add_argument("--params", help="Parámetros JSON (usa los que ganaron en el backtest)")
    p.add_argument("--exit-mode", choices=["premium", "underlying"], default="premium",
                   help="premium: stop y escalones sobre la prima | underlying: stop/objetivo en el precio de SPY")
    p.add_argument("--stop-pct", type=float, default=25, help="Stop sobre la prima en %% (modo premium)")
    p.add_argument("--tiers", default="25:50,50:50",
                   help="Escalones ganancia%%:contratos%%, ej. '25:50,50:50' = mitad en +25%%, mitad en +50%%")
    p.add_argument("--breakeven", action=argparse.BooleanOptionalAction, default=True,
                   help="Subir el stop a breakeven después del primer escalón")
    p.add_argument("--dte", type=int, default=1, help="Días mínimos al vencimiento (0 = vence hoy)")
    p.add_argument("--delta", type=float, default=0.65, help="Delta buscado (0.45 ≈ ATM, 0.65 = ITM)")
    p.add_argument("--max-minutes", type=int, default=30, help="Cerrar si no funcionó en N minutos (0 = sin límite)")
    p.add_argument("--feed", choices=["tradier", "ibkr"], default="tradier",
                   help="De dónde salen las velas y precios de opciones (las órdenes siempre van a IBKR)")
    p.add_argument("--until", default="12:00", help="Hora ET límite para buscar entrada")
    p.add_argument("--poll", type=float, default=15, help="Segundos entre revisiones")
    p.add_argument("--send", action="store_true", help="Enviar avisos a Telegram")
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("paper-journal", help="Resumen de la bitácora de paper")
    p.set_defaults(func=cmd_paper_journal)

    p = sub.add_parser("telegram-test", help="Probar Telegram")
    p.set_defaults(func=cmd_telegram_test)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
