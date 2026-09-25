"""Uso: python -m market_signals <comando> [opciones]

Comandos:
  premarket   Reporte antes de la apertura (sesgo del día y niveles)
  monitor     Vigila una estrategia en vivo y alerta la señal
  download    Descarga barras de 1 minuto de Tradier a data/
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
        params = json.loads(args.params) if args.params else {}
        trades = bt.run_strategy(contexts, args.strategy, params, args.cost_r)
        print(f"== {args.strategy} {params} ==")
        for k, v in bt.summarize(trades).items():
            print(f"  {k:22s} {v}")
        print("\nResultados por mes (R neto):")
        print(bt.monthly_table(trades).to_string())
        for title, table in bt.breakdown(trades).items():
            print(f"\n{title}:\n{table.to_string()}")
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
    params = json.loads(args.params) if args.params else {}
    run_monitor(_client(settings), settings, notify, args.symbol.upper(), args.strategy, params,
                until=args.until, poll_seconds=args.poll)


def cmd_download(args) -> None:
    from .data import download_recent
    settings = load_settings()
    path = download_recent(_client(settings), args.symbol, days=args.days)
    print(f"Guardado en {path}")


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

    p = sub.add_parser("telegram-test", help="Probar Telegram")
    p.set_defaults(func=cmd_telegram_test)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
