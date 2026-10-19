# mean-reversion-bot — V12

This repository contains only the active V12 forward-validation bot. Historical selection/sizing scripts and stopped forward versions are intentionally not kept in the working tree.

## Frozen portfolio

| Component | Side | Risk/trade |
| --- | --- | ---: |
| BTC/USDT:USDT | LONG | 0.75% |
| BNB/USDT:USDT | LONG | 0.50% |
| DOT/USDT:USDT | SHORT | 0.75% |
| ADA/USDT:USDT | SHORT | 0.50% |

Portfolio constraints are code-frozen:

- max position: 50%
- max aggregate planned risk: 1.50%
- max concurrent positions: 3
- Demo only (`USE_TESTNET=True`)

Signal/execution contract:

- timeframe: 1h
- z-score: 20 / 2.2
- normalized 1h shock: 2.175 / 100
- regime: 1h SMA200 + 4h EMA200
- ATR: 14; stop: 1.5 ATR
- RR: 1.8
- pullback LIMIT: 3 bp; wait 7 min; no-chase cap 5 bp
- LONG and SHORT require z-score and shock confirmation on the same closed candle

## Setup

```bash
cp .env.example .env
```

Set the Binance Demo credentials and set `V12_START_UTC` **once** to the first full 1h candle after deployment. Never move that boundary after the forward starts.

The runtime uses clean generic files:

- `state/forward.jsonl`
- `state/execution_journal.jsonl`
- `state/bot_state_<symbol>.json`

Market-data caches live below `state/history`, `state/minute` and `state/funding`. They are reusable inputs, not experiment outputs.

## Run

```bash
python -m pytest -q
python main.py
```

The bot fails closed if the frozen contract drifts, if Demo mode is disabled, or if `V12_START_UTC` is missing/invalid.

## Forward report

```bash
python forward_report.py
```

The report replays the frozen V12 signal on mainnet market data from `V12_START_UTC`, reports each component, a BTC+BNB baseline, the complete V12 portfolio, and Demo execution diagnostics. Demo fills/slippage are operational diagnostics only; they are not evidence of real mainnet execution quality.

## State hygiene

Generated research/old-forward artifacts are intentionally ignored by Git. When migrating to this clean tree, remove old generated files while keeping only the reusable market-data caches. The patch release instructions provide the exact cleanup command.
