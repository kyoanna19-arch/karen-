"""Uso: python -m market_signals <comando> [opciones]

Comandos:
  premarket   Reporte antes de la apertura (sesgo del día y niveles)
  monitor     Vigila una estrategia en vivo y alerta la señal
  download    Descarga barras de 1 minuto de Tradier a data/
  history     Descarga años de historial de 1 minuto desde Polygon.io
  options-study  Estudia el stop sobre la prima y las salidas escalonadas
  daily-history  Descarga velas diarias de muchos años (Yahoo Finance, gratis)
  swing       Prueba estrategias de swing (varios días) en velas diarias
  swing-portfolio  Simula el portafolio de swing con tu cuenta (varios activos, acciones enteras)
  swing-scan  Escáner diario: corre después del cierre, avisa y (opcional) manda órdenes a IBKR paper
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


def cmd_compare(args) -> None:
    from .compare import compare
    settings = load_settings()
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    risk = args.risk or settings.account_size * settings.risk_per_trade_pct / 100
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Estrategia {args.strategy} {parse_params(args.params) or '(configuración por defecto)'}")
    print(f"Opción: {args.dte} días al vencimiento, delta {args.delta} | salida por niveles del activo")
    print(f"Cuenta ${settings.account_size:,.0f}, riesgo ${risk:,.0f} por trade, "
          f"tope {settings.max_position_pct:.0f}% en primas\n")
    table, missing = compare(symbols, strategy=args.strategy, params=parse_params(args.params), dte=args.dte,
                             delta=args.delta, risk=risk, account=settings.account_size,
                             max_position_pct=settings.max_position_pct, iv_override=args.iv,
                             slip_override=args.slip)
    if missing:
        print(f"⚠️ Sin datos (descárgalos con 'history'): {', '.join(missing)}\n")
    if table.empty:
        return
    print(table.to_string(index=False))
    print("""
Cómo leerlo:
  estructura = CALL: solo el CALL ITM | spread objetivo: CALL ITM + vender el strike del objetivo
               spread ATM: CALL ITM + vender el strike más cercano al precio
  ETF_avg_r  = resultado por trade operando el activo directo (ya con costos)
  avg_r      = resultado por trade con esa estructura, saliendo por los niveles del activo
  avg_r_AÑO  = lo mismo separado por año: si es positivo en los dos años, es más confiable
  costo_contrato / cabe = cuánto cuesta 1 contrato y si entra en tu tope de primas
Una ventaja real debería salir positiva en VARIOS activos, no solo en uno.""")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparacion.csv", index=False)


def cmd_daily_history(args) -> None:
    from .swing import download_daily
    for symbol in [x.strip().upper() for x in args.symbol.split(",") if x.strip()]:
        try:
            path = download_daily(symbol, start=args.start)
            print(f"  {symbol}: listo ({path})")
        except Exception as exc:  # noqa: BLE001 - mostrar el error y seguir con los demás
            print(f"  {symbol}: error: {exc}")


def cmd_swing(args) -> None:
    from . import swing
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    rule = swing.SwingRule(args.rule, args.entry, args.exit, args.max_days, not args.no_trend)
    print(f"Regla: {rule.label()}")
    print("Compra en la apertura siguiente a la señal y vende en la apertura siguiente a la salida.\n")
    table, missing = swing.compare_symbols(symbols, rule)
    if missing:
        print(f"⚠️ Sin datos (descárgalos con 'daily-history'): {', '.join(missing)}\n")
    if not table.empty:
        print("== 1. La regla en cada activo ==")
        print(table.to_string(index=False))
    detail = args.detail.upper()
    path = swing.daily_path(detail)
    if path.exists():
        df = swing.load_daily(path)
        print(f"\n== 2. Variantes de la regla en {detail} (¿es estable o depende de un número exacto?) ==")
        print(swing.grid_table(df, detail).to_string(index=False))
        trades, _ = swing.backtest(df, rule, swing.cost_for(detail))
        print(f"\n== 3. Año por año en {detail} ==")
        print(swing.yearly(trades).to_string())
        last = df["close"].iloc[-1]
        print(f"\nCon $3,000 alcanzan {swing.shares_for(3000, last)} acciones de {detail} (a ${last:,.2f}).")
    print("""
Cómo leerlo:
  prom_%            = ganancia promedio por trade (ya con costos)
  CAGR_%            = rendimiento anual compuesto usando toda la cuenta en cada trade
  caída_máx_%       = la peor caída de la cuenta desde un máximo
  tiempo_invertido_% = qué parte del tiempo tuviste dinero en el mercado
  comprar_y_mantener = comparar contra solo comprar y no vender nunca
  prom_%_1a/2a_mitad = si funciona en la primera Y en la segunda mitad de los años, es más confiable
Una ventaja real debería ser positiva en varios activos y en las dos mitades.""")


def _swing_cfg(args):
    from .swing import SwingRule
    from .swing_portfolio import PortfolioConfig
    return PortfolioConfig(rule=SwingRule(args.rule, args.entry, args.exit, args.max_days),
                           max_positions=args.max_positions, commission=args.commission, whole_shares=True)


def cmd_swing_portfolio(args) -> None:
    from dataclasses import replace
    from . import swing_portfolio as sp
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    ind, missing = sp.load_indicators(symbols)
    if missing:
        print(f"⚠️ Sin datos (descárgalos con 'daily-history'): {', '.join(missing)}\n")
    if not ind:
        return
    bench = ind.get("SPY")
    cfg = _swing_cfg(args)
    print(f"Activos: {', '.join(ind)} | cuenta ${args.account:,.0f} | comisión ${args.commission} por orden")
    print(f"Regla: {cfg.rule.label()}\n")
    print("== 1. ¿Cuántas posiciones a la vez? (acciones enteras, toda la historia) ==")
    rows = []
    for n in (1, 2, 3, 4):
        c = replace(cfg, max_positions=n)
        state, equity = sp.run_backtest(ind, c, args.account)
        rows.append({"posiciones_máx": n, "$_por_espacio": round(args.account / n),
                     **sp.portfolio_stats(state, equity, args.account, bench)})
    print(pd.DataFrame(rows).to_string(index=False))
    state, equity = sp.run_backtest(ind, cfg, args.account)
    print(f"\n== 2. Año por año con {cfg.max_positions} posiciones ==")
    print(sp.yearly_returns(equity).to_string())
    by_sym = pd.DataFrame(state.closed).groupby("symbol").agg(trades=("ret", "size"), ganadores=("ret", lambda r: round((r > 0).mean(), 2)),
                                                              prom_pct=("ret", lambda r: round(r.mean() * 100, 2)),
                                                              ganancia_total=("pnl", "sum"))
    print(f"\n== 3. Por activo ==\n{by_sym.round(2).to_string()}")
    print("""
Cómo leerlo:
  CAGR_%       = rendimiento anual compuesto de la cuenta (con acciones enteras y comisiones)
  caída_máx_%  = la peor caída de la cuenta desde un máximo
  SPY_mantener = comparar contra comprar SPY y no vender nunca""")


def cmd_swing_scan(args) -> None:
    from datetime import date
    from . import swing_portfolio as sp
    from .swing import download_daily
    settings = load_settings()
    notify = Notifier(settings.telegram_bot_token, settings.telegram_chat_id, args.send)
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    if not args.no_download:
        for sym in symbols:
            try:
                download_daily(sym, start="2015-01-01")
            except Exception as exc:  # noqa: BLE001
                print(f"  {sym}: no se pudo actualizar ({exc})")
    ind, missing = sp.load_indicators(symbols)
    if missing:
        print(f"⚠️ Sin datos: {', '.join(missing)}")
    cfg = _swing_cfg(args)
    state_path = Path(args.state)
    state = sp.load_state(state_path, args.account)
    last_data = max(df.index[-1] for df in ind.values()).date()
    if last_data < date.today() and date.today().weekday() < 5:
        print(f"⚠️ El último dato es del {last_data}. Si hoy hubo mercado, corre esto después de las 4:15 pm ET.")
    results = sp.scan(state, ind, cfg)
    if len(results) > 1:
        print(f"⚠️ Se procesaron {len(results)} días de una vez (¿no corriste el escáner algún día?). "
              "Las órdenes de esos días se simularon, pero no se mandaron a IBKR.")
    lines = [f"📈 SWING {state.last_date}"]
    for res in results:
        for f in res["filled"]:
            if f["side"] == "BUY":
                lines.append(f"✅ Compra ejecutada {res['date']}: {f['qty']:g} {f['symbol']} a ${f['price']}")
            else:
                lines.append(f"💵 Venta ejecutada {res['date']}: {f['qty']:g} {f['symbol']} a ${f['exit_price']} "
                             f"({f['ret'] * 100:+.2f}%, ${f['pnl']:+,.2f})")
        sp.append_journal([f for f in res["filled"] if f["side"] == "SELL"], Path(args.journal))
    today_orders = results[-1]["orders"] if results else []
    for o in today_orders:
        verb = "COMPRAR" if o.side == "BUY" else "VENDER"
        lines.append(f"➡️ Mañana en la apertura: {verb} {o.qty:g} {o.symbol}")
    if not today_orders:
        lines.append("Sin órdenes nuevas para mañana.")
    last = pd.Timestamp(state.last_date)
    for sym, pos in state.positions.items():
        px = ind[sym].loc[:last, "close"].iloc[-1]
        lines.append(f"📌 Tienes {pos.qty:g} {sym} desde {pos.entry_date} a ${pos.entry_price:.2f} "
                     f"(hoy ${px:.2f}, {(px / pos.entry_price - 1) * 100:+.2f}%)")
    lines.append(f"Capital: ${state.equity(ind, last):,.2f} (efectivo ${state.cash:,.2f})")
    if args.execute and today_orders and len(results) == 1:
        broker = _ibkr(client_id=9)
        try:
            for o in today_orders:
                broker.place_moo(o.symbol, o.side, int(o.qty))
                lines.append(f"🧪 Orden MOO enviada a IBKR paper: {o.side} {int(o.qty)} {o.symbol}")
            ib_pos = broker.stock_positions()
            mine = {k: v.qty for k, v in state.positions.items()}
            if {k: v for k, v in ib_pos.items() if k in symbols} != mine:
                lines.append(f"⚠️ IBKR tiene {ib_pos} y la bitácora local {mine}. Revisa en TWS.")
        finally:
            broker.disconnect()
    sp.save_state(state, state_path)
    notify("\n".join(lines))


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

    study = osim.stop_study(pairs, iv=args.iv, slip=args.slip, delta=args.delta, dte=args.dte)
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

    print("\n== 8. Salir por los niveles del ETF (stop y objetivo de la estrategia, como en el backtest) ==")
    print("1R = lo que pierde la opción si el ETF toca el stop. Compara avg_r con las secciones anteriores.")

    def run_levels(dte, delta, minutes=None):
        trades = [t for t in (osim.simulate_option_levels(c, s, args.strategy, iv=args.iv, delta=delta, dte=dte,
                                                          slip=args.slip, max_minutes=minutes) for c, s in pairs) if t]
        st = bt.summarize(trades)
        return trades, {"trades": st["trades"], "win_rate": st["win_rate"], "avg_r": st["avg_r"],
                        f"$_por_trade_riesgo_{args.risk:.0f}": round(st["avg_r"] * args.risk, 2),
                        "total_r": st["total_r"], "max_dd_r": st["max_drawdown_r"],
                        "meses_+": st["pct_months_positive"], "peor_mes_r": st["worst_month_r"]}

    rows = []
    for dte, delta in ((0, 0.45), (0, 0.70), (1, 0.65), (2, 0.70), (5, 0.70)):
        for minutes in (None, 30):
            rows.append({"DTE": dte, "delta": delta, "máx_min": minutes or "-", **run_levels(dte, delta, minutes)[1]})
    print(pd.DataFrame(rows).to_string(index=False))
    level_trades = run_levels(args.dte, args.delta)[0]
    print(f"\nMes por mes saliendo por niveles del ETF ({args.dte}DTE delta {args.delta}):")
    print(bt.monthly_table(level_trades).to_string())

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

    p = sub.add_parser("daily-history", help="Descargar velas diarias de muchos años (Yahoo Finance)")
    p.add_argument("--symbol", default="SPY,QQQ,IWM,DIA,NVDA,AAPL,META,TSLA,MSFT,AMD")
    p.add_argument("--start", default="2005-01-01")
    p.set_defaults(func=cmd_daily_history)

    p = sub.add_parser("swing", help="Probar estrategias de swing en velas diarias")
    p.add_argument("--symbols", default="SPY,QQQ,NVDA,AAPL,META,TSLA,MSFT,AMD")
    p.add_argument("--rule", choices=["rsi2", "down3", "ibs"], default="rsi2")
    p.add_argument("--entry", type=float, default=10, help="Nivel de entrada (RSI2 < 10, o IBS < 20)")
    p.add_argument("--exit", choices=["sma5", "rsi70"], default="sma5")
    p.add_argument("--max-days", type=int, default=10)
    p.add_argument("--no-trend", action="store_true", help="Quitar el filtro del promedio de 200 días")
    p.add_argument("--detail", default="SPY", help="Activo para ver variantes y año por año")
    p.set_defaults(func=cmd_swing)

    def swing_args(p):
        p.add_argument("--symbols", default="SPY,QQQ,IWM,DIA")
        p.add_argument("--account", type=float, default=3000)
        p.add_argument("--max-positions", type=int, default=2)
        p.add_argument("--commission", type=float, default=1.0, help="Dólares por orden")
        p.add_argument("--rule", choices=["rsi2", "down3", "ibs"], default="rsi2")
        p.add_argument("--entry", type=float, default=10)
        p.add_argument("--exit", choices=["sma5", "rsi70"], default="sma5")
        p.add_argument("--max-days", type=int, default=10)

    p = sub.add_parser("swing-portfolio", help="Simular el portafolio de swing con tu cuenta")
    swing_args(p)
    p.set_defaults(func=cmd_swing_portfolio)

    p = sub.add_parser("swing-scan", help="Escáner diario de swing (correr después del cierre)")
    swing_args(p)
    p.add_argument("--state", default="data/swing_state.json", help="Archivo con tus posiciones simuladas")
    p.add_argument("--journal", default="data/swing_journal.csv")
    p.add_argument("--no-download", action="store_true", help="No actualizar datos de Yahoo")
    p.add_argument("--execute", action="store_true", help="Mandar las órdenes a IBKR paper (MOO)")
    p.add_argument("--send", action="store_true", help="Enviar el resumen a Telegram")
    p.set_defaults(func=cmd_swing_scan)

    p = sub.add_parser("compare", help="Probar la misma estrategia en varios activos (una sola tabla)")
    p.add_argument("--symbols", default="SPY,QQQ,NVDA,AAPL,META,TSLA,MSFT,AMD")
    p.add_argument("--strategy", choices=list(STRATEGIES), default="karen_pullback")
    p.add_argument("--params", help="Parámetros de la estrategia (por defecto, la configuración elegida)")
    p.add_argument("--dte", type=int, default=2, help="Días al vencimiento de la opción")
    p.add_argument("--delta", type=float, default=0.70, help="Delta de la opción")
    p.add_argument("--risk", type=float, help="Dólares en riesgo por trade (por defecto, el de tu .env)")
    p.add_argument("--iv", type=float, help="Forzar una misma volatilidad para todos (por defecto, la típica de cada uno)")
    p.add_argument("--slip", type=float, help="Forzar un mismo costo por lado (por defecto 0.02 ETFs, 0.05 acciones)")
    p.add_argument("--out", default="reports")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("options-study", help="Estudiar stop sobre la prima y salidas escalonadas")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV de barras de 1 minuto")
    src.add_argument("--demo", action="store_true", help="Usar datos sintéticos de prueba")
    p.add_argument("--demo-days", type=int, default=250)
    p.add_argument("--strategy", choices=list(STRATEGIES), default="karen_pullback")
    p.add_argument("--params", help="Parámetros JSON de la estrategia")
    p.add_argument("--stop", type=float, default=0.25, help="Stop sobre la prima (0.25 = 25%%)")
    p.add_argument("--iv", type=float, default=0.18, help="Volatilidad implícita supuesta (0.18 = 18%%)")
    p.add_argument("--dte", type=int, default=0, help="Días al vencimiento de la opción (0 = vence hoy)")
    p.add_argument("--delta", type=float, default=0.45, help="Delta de la opción (0.45 ≈ ATM, 0.70 = ITM)")
    p.add_argument("--slip", type=float, default=0.02,
                   help="Costo por lado en dólares: medio spread + deslizamiento (0.02 = 2 centavos)")
    p.add_argument("--risk", type=float, default=80, help="Dólares en riesgo por trade")
    p.add_argument("--out", default="reports")
    p.set_defaults(func=cmd_options_study)

    p = sub.add_parser("ibkr-test", help="Probar la conexión con IBKR paper")
    p.add_argument("--symbol", default="SPY")
    p.set_defaults(func=cmd_ibkr_test)

    p = sub.add_parser("paper", help="Operar automáticamente en la cuenta PAPER de IBKR")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--strategy", choices=list(STRATEGIES), default="karen_pullback")
    p.add_argument("--params", help="Parámetros JSON (usa los que ganaron en el backtest)")
    p.add_argument("--exit-mode", choices=["underlying", "premium"], default="underlying",
                   help="underlying: stop/objetivo en el precio del ETF (como el backtest) | "
                        "premium: stop y escalones sobre la prima")
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
