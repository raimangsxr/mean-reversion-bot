# mean-reversion-bot — V9.2

Bot BTC mean-reversion para Binance USD-M Futures. V9.2 congela las conclusiones de V9.1 y deja de retunear la estrategia por ano.

## Configuracion live/Demo V9.2

La variante promovida a Demo es estatica:

- BTC/USDT:USDT, LONG only;
- entrada confirmed z-score + 1h SMA200 + 4h EMA200;
- LIMIT postOnly a 3 bp del signal close;
- espera 7 minutos;
- fallback MARKET solo si sigue dentro de +5 bp respecto a t=0;
- SL = 1.5 ATR desde el fill real;
- TP = fixed RR 1.8, LIMIT reduceOnly postOnly;
- `RISK_PER_TRADE=0.005`;
- `MAX_POSITION_PCT=0.30`;
- isolated, leverage 2.

Las variantes mean/BE/parciales/time-stop y SHORT siguen soportadas por el codigo de research/live, pero no forman parte del grid V9.2 por defecto porque V9.1 no demostro mejora frente a `fixed_rr + LONG`.

## Seleccion V9.2

El objetivo es maximizar return sujeto a una frontera de robustez. Ya no se exige 5/5 anos positivos como condicion binaria.

Defaults:

- max DD <= 10%;
- al menos 4 anos positivos;
- peor PF anual >= 0.90;
- peor return anual >= -1%;
- PF global >= 1.20.

Dentro de esa frontera, gana el mayor return. Con los resultados V9.1 suministrados, este criterio selecciona:

`fixed_rr | RR=1.8 | risk=0.5% | max-position=30% | LONG only`.

El anchored walk-forward dinamico se conserva exclusivamente como diagnostico de retuning. No se usa para elegir ni cambiar automaticamente la configuracion live.

## Research principal

Por defecto el research es BTC-only:

```dotenv
BACKTEST_SYMBOLS=BTC/USDT:USDT
BACKTEST_SELECTION_SYMBOLS=BTC/USDT:USDT
BACKTEST_EXIT_PROFILES=fixed_rr
BACKTEST_TP_RATIOS=1.5,1.8
BACKTEST_MAX_POSITION_PCT_GRID=0.10,0.30,0.50,0.60
BACKTEST_SIDE_MODES=long_only
```

ETH/SOL pueden anadirse manualmente para experimentos separados, pero ya no se descargan ni calculan en cada backtest normal.

## Realismo de ejecucion

El backtest incluye:

- fees maker/taker configurables;
- slippage de mercado;
- funding historico por timestamp;
- TP maker solo si el mercado cruza el objetivo el numero configurado de ticks;
- SL primero si una OHLC toca SL y TP;
- no usar el high previo al fill LIMIT dentro del mismo minuto;
- cierre MARKET al final del dataset si queda una posicion abierta;
- sizing compartido por equity/margen/riesgo.

## Sensibilidad y stress

El grid completo se selecciona al slippage primario (2 bp). Los finalistas se vuelven a medir por defecto en 1/2/3/5 bp.

El candidato final ademas se somete a stress sin reabrir la optimizacion:

- fees x1.0 / x1.2;
- funding x1.0 / x1.5 / x2.0;
- TP maker con cruce de 1 / 2 ticks.

Salida adicional:

`state/backtest_v9_stress.csv`.

## Scheduler y seguridad live

Cuando esta flat, el proceso se alinea al cierre de la vela (`HH:00:02`) en lugar de acumular `sleep(60)`. Las entradas pending se reconcilian cada 10 s. El motor mantiene persistencia JSON, reconciliacion fail-closed y proteccion SL/TP desde el fill real.

## Caches

- `state/minute/`: ventanas 1m;
- `state/funding/`: funding historico;
- `state/history/`: 1h/4h incremental;
- `state/backtest_v9_opportunities.csv`: oportunidades unitarias cacheadas;
- `.meta.json`: fingerprint de inputs economicos.

Para forzar regeneracion:

```bash
BACKTEST_REBUILD_OPPORTUNITIES=True python backtest.py
```

## Ejecutar

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python backtest.py
```

Demo/live:

```bash
caffeinate -i python main.py
```

## Outputs

- `state/backtest_v9_summary.csv`
- `state/backtest_v9_annual.csv`
- `state/backtest_v9_walk_forward.csv`
- `state/backtest_v9_walk_forward_trades.csv`
- `state/backtest_v9_sensitivity.csv`
- `state/backtest_v9_stress.csv`
- `state/backtest_v9_opportunities.csv`
- `state/backtest_v9_trades.csv`
- `state/backtest_v9_recommendation.json`

`recommendation.json` incluye el candidato estatico, su detalle anual, stress y el walk-forward dinamico solo como diagnostico.
