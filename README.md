# mean-reversion-bot — V8 final

Bot LONG de mean reversion para Binance USD-M Futures (`BTC/USDT:USDT`). Esta
rama contiene solo la estrategia y la ejecucion V8 final.

## Estrategia

- LONG only.
- Senal sobre velas 1h cerradas.
- Entrada `confirmed`: recuperacion del z-score desde debajo de `-2.2`.
- z-score con ventana 20.
- Regimen: `close1h > SMA200_1h` y `close4h > EMA200_4h`.
- ATR(14), stop inicial `1.5 ATR`.
- RR live actual: `1.5` (el backtest final compara tambien `1.8`).
- SL: `STOP_MARKET reduceOnly` (taker).
- TP: `LIMIT reduceOnly + postOnly` (maker).

## Entrada V8 final: pullback / no-chase

La ejecucion usa dos referencias:

1. `limit_reference`: cierre de la vela 1h que genero la senal.
2. `t0_reference`: precio de mercado observado al iniciar la ejecucion.

Para LONG:

1. Se intenta `BUY LIMIT postOnly` a `limit_reference - 3 bp`.
2. Se mantiene pendiente durante 7 minutos.
3. Si hay fill parcial, se cancela el remanente y se protege inmediatamente la
   cantidad ejecutada.
4. Si la LIMIT no puede descansar como post-only, se entra en modo `wait-only`;
   no se fuerza un fill taker.
5. Al expirar, solo se permite fallback MARKET si
   `current_price <= t0_reference + 5 bp`.
6. Si el precio supera el cap, la senal se consume y se cancela.
7. Si el proceso recupera la entrada mas de 60 s tarde respecto a su expiracion,
   tampoco ejecuta el fallback.

SL y TP se calculan desde el fill real.

## Persistencia y seguridad

El estado de una entrada pendiente se persiste en `STATE_FILE`, de modo que un
reinicio puede reconciliar la orden sin duplicarla. El motor trabaja fail-closed:
si no puede verificar posicion u orden, no asume que la cuenta esta flat.

Los ficheros de `state/` son runtime y no se versionan.

## Backtest V8 final

`backtest.py` compara:

- `market_0m`: control con entrada MARKET inmediata y TP maker.
- `pullback_limit`: entrada V8 final descrita arriba.

Barre RR `1.5/1.8` y el grid de slippage configurado en `.env`. Usa datos 1m solo
en las horas posteriores a senales candidatas y los cachea en
`state/backtest_1m_cache.csv`.

Genera:

- `state/backtest_v8_summary.csv`
- `state/backtest_v8_annual.csv`
- `state/backtest_v8_trades.csv`

La seleccion final de RR prioriza robustez anual, peor PF anual, PF mediano,
PF global y drawdown; no solo el mejor resultado agregado.

## Configuracion

Copia `.env.example` a `.env` y completa las credenciales de Binance Demo.
`.env` esta excluido de Git y del contexto Docker.

## Comandos

Instalacion:

```bash
python -m pip install -r requirements.txt
```

Tests:

```bash
python -m pytest -q
```

Backtest:

```bash
python backtest.py
```

Demo/live en macOS evitando suspension:

```bash
caffeinate -i python main.py
```

Docker:

```bash
docker compose up -d --build
```

El contenedor persiste unicamente `./state` como volumen; el codigo se incorpora
a la imagen durante el build.
