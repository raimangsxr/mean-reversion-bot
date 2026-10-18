# Migration to V10.2

V10.2 is a repository consolidation, not a live strategy migration.

Removed source entrypoints:

- `backtest.py`
- `backtest_v10.py`
- `backtest_v10_1.py`

Renamed internal research modules:

- `src/research/v9.py` -> `src/research/engine.py`
- `src/research/v10.py` -> `src/research/signals.py`

Removed old migration documents. V10.2 is now the only research version represented in the repository.

The first `python backtest_v10_2.py` run detects a legacy funding CSV without snapshot metadata and rebuilds it once. Subsequent runs verify SHA-256 and append only new funding rows.

Generated research files from older versions are deleted automatically when `V10_2_CLEAN_OLD_ARTIFACTS=True`; reusable market-data caches and live state are preserved.
