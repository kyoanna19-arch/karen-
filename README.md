# Sistema de señales para scalping de opciones (EE.UU.)

El sistema hace tres cosas:

1. **Te prepara el día.** Antes de la apertura genera un reporte con el sesgo del mercado (alcista, bajista o neutral) y los niveles clave de SPY, QQQ e IWM.
2. **Descubre qué estrategia funciona de verdad.** Prueba varias estrategias con datos históricos, **mes por mes**, y verifica que no sea suerte.
3. **Te avisa en Telegram** cuando aparece la señal. Te sugiere el contrato de opción y cuántos contratos comprar según tu riesgo.

> No envía órdenes. Tú decides siempre. TC2000 sigue siendo tu pantalla para confirmar visualmente.

---

## 1. Instalación (una sola vez)

Necesitas Python 3.10 o superior.

```bash
git clone <este repo>
cd karen-
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # luego edita .env con tus datos
```

### Token de Tradier
1. Entra a https://dash.tradier.com/settings/api
2. Copia el **Sandbox token** (para practicar) y ponlo en `TRADIER_TOKEN` con `TRADIER_ENV=sandbox`.
3. Cuando quieras datos en tiempo real, usa tu token de producción con `TRADIER_ENV=live`.
   El sandbox da datos con retraso de ~15 minutos.

### Bot de Telegram
1. En Telegram, habla con **@BotFather**, escribe `/newbot` y copia el token en `TELEGRAM_BOT_TOKEN`.
2. Mándale cualquier mensaje a tu bot nuevo.
3. Habla con **@userinfobot** para ver tu número de chat y ponlo en `TELEGRAM_CHAT_ID`.
4. Prueba: `python -m market_signals telegram-test`

**Nunca compartas ni subas tu archivo `.env`.** Ya está excluido de git.

---

## 2. Rutina diaria (hora del Este, ET)

| Hora ET | Comando | Qué hace |
|---|---|---|
| 9:00 | `python -m market_signals premarket --send` | Reporte con sesgo y niveles a Telegram |
| 9:30 | `python -m market_signals monitor --symbol SPY --strategy orb --params '{"range_minutes":5,"target_r":1.5}' --send` | Vigila la estrategia hasta las 11:30 y te avisa |
| Viernes | `python -m market_signals download --symbol SPY` | Guarda las barras de 1 minuto de la semana |

En el monitor, usa la estrategia y los parámetros que **ganaron en el backtest**, no los del ejemplo.

---

## 3. Encontrar la mejor estrategia (backtest)

```bash
# Probar que todo corre (datos falsos, los números no significan nada)
python -m market_signals backtest --demo

# Con tus datos reales
python -m market_signals backtest --data data/SPY_1min.csv

# Análisis detallado de una estrategia
python -m market_signals backtest --data data/SPY_1min.csv --strategy orb --params '{"range_minutes":15,"target_r":1.5}'
```

### Estrategias incluidas

| Nombre | Idea |
|---|---|
| `orb` | Ruptura del rango de los primeros 5 o 15 minutos |
| `premarket_break` | El precio cruza el máximo o mínimo del pre-market |
| `vwap_cross` | Recupera o pierde el VWAP con las EMAs 9/20 a favor |
| `gap_fade` | Apuesta a que el gap de apertura se rellena |
| `ema_vwap_rsi` | **Tu estrategia**: EMA 9/21, vela sólida que rompe los promedios, VWAP girado, RSI 60/40 y volumen |

### Tu estrategia traducida a reglas exactas

**CALL** (PUT es el espejo, con RSI < 40):
1. EMA 9 > EMA 21, y **las dos** subiendo respecto a la vela anterior.
2. **Vela sólida:** el cuerpo mide al menos el 60% del rango de la vela (`body_min=0.6`), es verde, su mínimo toca la EMA 9 y **cierra arriba de las dos EMAs**.
3. **Sobre el VWAP y VWAP girado:** el cierre está arriba del VWAP y el VWAP está más alto que hace 3 velas. La distancia al VWAP se limita con `max_vwap_dist_pct`; el backtest prueba 0.3% y "sin límite".
4. **RSI(14) > 60.**
5. **Volumen:** mayor que la vela anterior (`vol_mode=prev`) o mayor que las 4 anteriores (`max4`); el backtest prueba las dos.

6. **Confirmación en temporalidad mayor (opcional, `confirm_tf`)**: en la última vela de 5 o 15 minutos **ya cerrada**, EMA 9 > EMA 21 y cierre sobre el VWAP. Es como mirar la gráfica de 15m para la tendencia y entrar en la de 1m o 5m.

Combinaciones de temporalidad que se prueban (las mismas que usas en TC2000):

| Entrada (`timeframe`) | Confirmación (`confirm_tf`) |
|---|---|
| 1 min | ninguna, 5 min o 15 min |
| 5 min | ninguna o 15 min |
| 15 min | ninguna |

Lo que no estaba en tus reglas y el backtest decide: stop en el extremo de la vela de señal o en la EMA 21, objetivo de 1R, 1.5R o 2R, entradas entre 9:35 y 11:30 y salida por tiempo a las 12:00.

```bash
# Buscar la mejor versión de tu estrategia
python -m market_signals backtest --data data/SPY_1min.csv --only ema_vwap_rsi

# Detalle mes por mes + qué regla aporta y cuál sobra
python -m market_signals backtest --data data/SPY_1min.csv --strategy ema_vwap_rsi --params '{"timeframe":1,"confirm_tf":15}'
```

La tabla **"¿Qué regla aporta?"** apaga una regla a la vez. Si al quitar una regla el resultado **mejora** o no cambia, esa regla no está sumando y solo te quita trades buenos. Si al quitarla **empeora**, es una regla importante.

⚠️ **Cuidado con pocos trades.** Con velas de 15 minutos hay pocas señales por mes. Un resultado con 20–30 trades puede ser pura suerte. Confía solo en lo que se sostiene en el walk-forward y con 100+ trades.

Cada una se prueba con varias combinaciones de parámetros. El filtro `with_gap` (operar solo a favor del gap) usa el sesgo del pre-market. Solo se toma **una señal por día** porque la regla PDT te limita.

### Cómo leer los resultados

- **R** es el múltiplo de tu riesgo. Si arriesgas $80 y ganas $160, ese trade fue +2R.
- **avg_r** es la ganancia promedio por trade en R, ya con costos descontados. Tiene que ser mayor que 0.
- **pct_months_positive** es el % de meses que terminaron en positivo. Esto responde a tu problema de que "no funciona todos los meses".
- **worst_month_r** es el peor mes. Te dice cuánto aguantar sin perder la disciplina.
- **Walk-forward** elige los parámetros con el 70% inicial de los datos y los prueba en el 30% final, que nunca vio. **Esta es la tabla que importa.** Si una estrategia solo brilla en entrenamiento, fue suerte.
- **Desglose por gap, día y dirección** te dice *cuándo* funciona una estrategia. Ninguna funciona siempre: la meta es saber qué días usar cuál y qué días no operar.

`cost_r=0.1` descuenta 0.1R por trade para cubrir spread y deslizamiento. En opciones de acciones individuales el spread es mayor; sube el costo con `--cost-r 0.2`.

### Datos: el punto más importante

Tradier solo guarda unas **semanas** de barras de 1 minuto. Para un backtest confiable necesitas **al menos 1 año**. Tienes dos opciones:

1. **Gratis, pero lento:** corre `download` cada semana y el archivo va creciendo.
2. **Rápido:** compra o descarga historial de 1 minuto de un proveedor (Polygon, Databento, FirstRate Data, etc.) y guárdalo como CSV con columnas `datetime,open,high,low,close,volume` en hora del Este, incluyendo el pre-market.

---

## 4. Gestión de riesgo con una cuenta de $8,000

- **El riesgo por trade viene configurado en 1% ($80).** El sistema calcula cuántos contratos puedes comprar si sales cuando la opción pierde `OPTION_STOP_PCT`% (30% por defecto). Si la prima es muy cara, te dice que no entres.
- **Regla PDT.** En una cuenta de margen con menos de $25k, FINRA limita a 3 day trades en 5 días hábiles. FINRA ha estado cambiando esta regla, así que **confirma con Tradier cómo te aplica hoy**. Una alternativa es una **cuenta cash**: las opciones liquidan en T+1, no aplica PDT, pero solo puedes operar con dinero ya liquidado.
- **Pérdida máxima diaria sugerida:** 2R ($160). Si la alcanzas, apaga todo hasta mañana.
- **Opciones 0DTE:** el theta es brutal después del mediodía. Por eso las estrategias salen por tiempo (11:30–11:45 ET).

---

## 5. Plan de trabajo

- [x] Fase 1: conexión a Tradier, reporte pre-market, alertas en Telegram
- [x] Fase 2: estrategias y backtest con validación walk-forward
- [ ] Juntar mínimo 1 año de datos de SPY y QQQ y correr el backtest real
- [ ] Traducir tus scans de TC2000 a estrategias nuevas en `market_signals/strategies.py`
- [ ] 4–6 semanas de paper trading (sandbox) siguiendo las alertas
- [ ] Filtro de régimen: elegir la estrategia del día según el sesgo del pre-market
- [ ] Contador automático de day trades (PDT) y bitácora de operaciones

---

## Estructura

```
market_signals/
  config.py      configuración (.env)
  tradier.py     cliente de la API de Tradier (solo lectura)
  telegram.py    envío de alertas
  premarket.py   reporte pre-market y sesgo del día
  strategies.py  estrategias (aquí se agregan las tuyas)
  backtest.py    simulación, estadísticas mensuales, optimización, walk-forward
  options.py     elección de contrato y tamaño de posición
  monitor.py     vigilancia en vivo
  data.py        descarga de barras
tests/           pruebas automáticas (python -m pytest)
```

*Esto es una herramienta educativa, no asesoría financiera. Operar opciones implica riesgo de perder todo lo invertido.*
