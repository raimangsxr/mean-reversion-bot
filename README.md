# mean-reversion-bot - V10.4

V10.4 is the only research version kept in the repository. Live/Demo execution remains unchanged while research studies one question: can the frozen `z-score + shock1h` edge be replicated across additional liquid USD-M perpetuals to raise trade frequency without destroying PF?

## Live / Demo

Live remains unchanged and is not auto-promoted by V10.4:

- BTC/USDT:USDT, LONG only;
- confirmed z-score re-entry, window 20 / threshold 2.2;
- regime 1h SMA200 + 4h EMA200;
- LIMIT postOnly at -3 bp, wait 7 minutes;
- MARKET fallback only inside the +5 bp no-chase cap;
- SL = 1.5 ATR from actual fill;
- TP = fixed RR 1.8, reduceOnly maker LIMIT;
- risk/trade = 0.5%; max position = 30%; isolated 2x.

## Frozen V10.4 research signal

Every market uses exactly the same 1h setup:

```text
confirmed z-score re-entry: z[t-1] < -2.2 and z[t] >= -2.2
AND
confirmed normalized 1-hour downside-shock re-entry at threshold 2.175
```

No threshold, timeframe, RR, ATR, regime, sizing or execution parameter is retuned per asset.

## Cross-asset universe

The reference portfolio from V10.3 is BTC + BNB. V10.4 adds a pre-registered expansion basket:

```text
BTC, BNB,
SOL, DOGE, ADA, LINK, AVAX, BCH, TRX, DOT
```

ETH/LTC/XRP are not in the default V10.4 basket because V10.3 already rejected them individually with the same frozen signal.

## Component gate

Every market must demonstrate edge independently before it can enter the expansion path:

```dotenv
V10_4_MIN_COMPONENT_TRADES=20
V10_4_MIN_COMPONENT_PF=1.20
V10_4_COMPONENT_DD_LIMIT=0.10
V10_4_MIN_POSITIVE_YEARS=3
```

## Portfolio construction rule

V10.4 deliberately avoids an exhaustive subset search.

1. Start from the individually-qualified reference symbols (BTC + BNB).
2. Rank only the newly-qualified assets by individual PF.
3. Add them one by one in that fixed ranking.
4. The **first** sequential step that reaches all portfolio gates is the historical forward-test candidate.
5. Later combinations are not allowed to replace it simply because they backtest better.

Portfolio gates:

```dotenv
V10_4_TARGET_MIN_TRADES_PER_YEAR=24
V10_4_TARGET_MAX_TRADES_PER_YEAR=40
V10_4_PORTFOLIO_MIN_PF=1.60
V10_4_PORTFOLIO_DD_LIMIT=0.07
V10_4_MIN_PORTFOLIO_POSITIVE_YEARS=4
```

V10.4 also reports trades/month plus median, P90 and maximum days between entries so frequency is measured as actual cadence, not just an annual average.

## Statistical status

The BTC signal thresholds were discovered after inspecting 2022+, so 2022+ is not a fresh temporal holdout. The new assets are useful cross-sectional external tests because their data was not used to derive those BTC thresholds, but any portfolio selected by V10.4 still requires fresh forward validation.

`promotion_to_live` therefore remains `false`.

## Funding reproducibility

Funding is snapshotted per symbol under `state/funding/` with SHA-256 metadata. Historical rows are immutable; later runs may append only new tail rows. Gaps or historical modifications fail closed.

To deliberately rebuild snapshots:

```bash
python backtest_v10_4.py --rebuild-funding
```

Do not use this casually if exact reproducibility matters.

## Run

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python backtest_v10_4.py
```

The first run may download 1h/4h history, funding and selected 1m execution windows for the new assets. Subsequent runs reuse caches.

## Outputs

Only V10.4 research artifacts are produced:

- `state/backtest_v10_4_markets.csv`
- `state/backtest_v10_4_expansion.csv`
- `state/backtest_v10_4_portfolios.csv`
- `state/backtest_v10_4_annual.csv`
- `state/backtest_v10_4_opportunities.csv`
- `state/backtest_v10_4_sensitivity.csv`
- `state/backtest_v10_4_stress.csv`
- `state/backtest_v10_4_recommendation.json`

Old V9/V10/V10.1/V10.2/V10.3 generated research artifacts are deleted automatically. Historical/minute/funding caches, live state and execution journals are preserved.
