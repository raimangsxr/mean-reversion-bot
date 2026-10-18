# mean-reversion-bot - V10.2

V10.2 consolidates the repository around one live code path and one research experiment.
Older research entrypoints and versioned modules have been removed.

## Live / Demo

The live trigger is intentionally unchanged while V10.2 is researched:

- BTC/USDT:USDT, LONG only;
- confirmed z-score re-entry: window 20, threshold 2.2;
- regime: 1h SMA200 + 4h EMA200;
- LIMIT postOnly at -3 bp from signal close, wait 7 minutes;
- MARKET fallback only inside the +5 bp no-chase cap;
- SL = 1.5 ATR from actual fill;
- TP = fixed RR 1.8, reduceOnly maker LIMIT;
- base risk/trade = 0.5%;
- max position = 30% of equity;
- isolated margin, leverage 2.

V10.2 does **not** switch the live signal to the new confirmation candidate automatically.

## V10.2 research question

The trigger remains z-score. V10.2 asks whether a same-candle confirmation improves trade quality:

1. `zscore_only` - control;
2. `zscore_and_shock1h` - primary hypothesis;
3. `zscore_and_shock3h`;
4. `zscore_and_vwap48h`;
5. `zscore_and_taker`.

The confirmation thresholds are frozen before this run:

- shock 1h = 2.175 normalized-vol units;
- shock 3h = 1.725;
- VWAP48 deviation = 1.55 ATR;
- taker exhaustion min shock = 0.925 ATR.

There is no threshold optimization in V10.2.

### Important statistical status

The shock1h combination was discovered after inspecting 2022+ results. Therefore V10.2 reports 2022+ as **development/diagnostic data**, not as a fresh holdout. Even if historical metrics are excellent, `recommendation.json` always keeps `promotion_to_live=false`; a fresh forward test is required.

## Funding reproducibility

V10.2 replaces mutable historical funding with a snapshot model:

- `state/funding/<symbol>.csv` contains rates;
- `state/funding/<symbol>.meta.json` stores row count, coverage and SHA-256;
- existing historical rows are immutable;
- later runs may only append new tail events;
- a gap greater than the configured limit fails closed;
- a manually changed historical CSV fails hash verification.

On the first V10.2 run, an old funding cache without metadata is rebuilt once automatically. To deliberately create a new snapshot later:

```bash
python backtest_v10_2.py --rebuild-funding
```

Do not use that flag casually if you want exact historical reproducibility.

## Sizing study

Only the primary `zscore_and_shock1h` candidate is subjected to a sizing grid. Defaults:

```dotenv
V10_2_RISK_GRID=0.005,0.0075,0.01,0.015
V10_2_MAX_POSITION_GRID=0.30,0.50,0.70,1.00
V10_2_SIZING_DD_LIMIT=0.10
V10_2_SIZING_MIN_PF=1.20
```

The output includes annualized historical return and an illustrative annual EUR amount on EUR 10,000. These are backtest statistics, not a forward expectation.

## Sensitivity and stress

The control and primary candidate are re-evaluated at 1/2/3/5 bp market slippage. Both also receive fee/funding stress using the configured multipliers.

## Cleanup

V10.2 removes old source entrypoints from the repository. By default the backtest also deletes generated files matching old research prefixes (`backtest_v9*`, `backtest_v10_*` except `backtest_v10_2_*`). It does **not** delete:

- live state or execution journals;
- `state/history/`;
- `state/minute/`;
- `state/taker/`;
- `state/funding/`.

Disable generated-artifact cleanup with:

```dotenv
V10_2_CLEAN_OLD_ARTIFACTS=False
```

## Run

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python backtest_v10_2.py
```

Demo/live remains:

```bash
caffeinate -i python main.py
```

## Outputs

- `state/backtest_v10_2_summary.csv`
- `state/backtest_v10_2_annual.csv`
- `state/backtest_v10_2_opportunities.csv`
- `state/backtest_v10_2_sizing.csv`
- `state/backtest_v10_2_sensitivity.csv`
- `state/backtest_v10_2_stress.csv`
- `state/backtest_v10_2_recommendation.json`

`backtest_v10_2_recommendation.json` records the funding snapshot hash, control/candidate metrics, historical sizing candidate, and the explicit requirement for fresh forward validation.
