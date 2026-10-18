# Migration to V10.4

V10.4 replaces V10.3 as the only research entrypoint in the repository.

## Removed

- `backtest_v10_3.py`
- `MIGRATION_V10_3.md`
- V10.3 multi-timeframe research/output configuration

Reusable history, minute and funding caches remain intact.

## Added

- `backtest_v10_4.py`
- fixed cross-asset 1h expansion basket: BTC, BNB, SOL, DOGE, ADA, LINK, AVAX, BCH, TRX, DOT;
- BTC+BNB as the frozen reference portfolio discovered in V10.3;
- the exact same `z-score 20/2.2 + shock1h 2.175` signal on every asset;
- mandatory individual component gate before portfolio inclusion;
- deterministic sequential portfolio expansion instead of subset optimization;
- portfolio frequency band 24-40 trades/year, PF >= 1.60, DD <= 7%, >= 4 positive years;
- cadence metrics: trades/month, median/P90/max gap between entries;
- slippage sensitivity and fee/funding stress on the selected/final deterministic expansion step.

## Construction rule

New individually-qualified markets are sorted by their own PF and added one at a time to the qualified BTC+BNB reference. The first step that satisfies all portfolio gates becomes the historical candidate. V10.4 does not search all subsets for the best backtest.

## Run

```bash
python -m pytest -q
python backtest_v10_4.py
```

Live remains unchanged. V10.4 is research-only and never auto-promotes to live.
