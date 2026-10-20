# mean-reversion-bot — V12

This repository contains the single active V12 forward-validation bot. Historical selection, sizing and consumed research runners are intentionally not kept in the working tree.

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
- execution: Binance USD-M Demo only (`USE_TESTNET=True`)

Signal contract:

- timeframe: 1h
- z-score: 20 / 2.2
- normalized 1h shock: 2.175 / 100
- regime: 1h SMA200 + 4h EMA200
- ATR: 14; stop: 1.5 ATR
- RR: 1.8
- LONG and SHORT require z-score and shock confirmation on the same closed candle

## Frozen order-flow execution gate

A valid 1h signal is not submitted immediately. It enters an `ARMED` state at the candle close and the bot observes public Binance USD-M mainnet `aggTrade` flow for exactly 60 seconds.

The signal is vetoed when both side-adjusted conditions hold at t+60s:

- aggressive-flow imbalance <= -20%
- price return <= -5 bp

This is the frozen `adverse_acceptance` condition. Otherwise the bot attempts the entry using:

- fresh pullback LIMIT: 3 bp from the last `aggTrade` price observed at t+60s
- total entry deadline: t0 + 7 minutes (the 60-second observation does not add another 7 minutes)
- no-chase fallback cap: 5 bp

The public market-data collector is fail-closed. If it cannot prove continuous WebSocket coverage for the complete 60-second window, the opportunity is discarded rather than traded on partial flow.

Orders remain Demo-only. Public signal/order-flow evidence is read from mainnet market-data streams so Demo liquidity does not define the strategy signal.

## Setup

```bash
cp .env.example .env
```

Set the Binance Demo credentials and set `V12_START_UTC` **once** to the first full 1h candle after deployment. Never move that boundary after the forward starts.

The runtime uses:

- `state/forward.jsonl` — signals, ARMED states, flow decisions and shadow references
- `state/execution_journal.jsonl` — Demo order/fill/position events
- `state/bot_state_<symbol>.json` — execution state

Market-data caches below `state/history`, `state/minute` and `state/funding` are reusable replay inputs rather than experiment output.

## Run

```bash
python -m pytest -q
python main.py
```

The bot fails closed if the frozen V12 contract drifts, Demo mode is disabled, `V12_START_UTC` is missing/invalid, or complete order-flow coverage cannot be established.

## Forward report

```bash
python forward_report.py
```

The report uses the live-observed forward decisions and compares three like-for-like paths over the fresh period:

- `baseline_shadow`: original V12 entry route without the 60-second delay;
- `delay_shadow`: always waits 60 seconds, but never applies the flow veto;
- `orderflow_shadow`: same delayed route, with the frozen `adverse_acceptance` veto.

It also reports Demo execution diagnostics from the journals. Demo fills/slippage are operational diagnostics only; the shadow comparison is the controlled test of whether order flow adds value beyond the delay itself.

Use `python forward_report.py --no-replay` for journal/execution diagnostics without refreshing public replay data.

## State hygiene

Generated artifacts are ignored by Git. Before the first V12 forward run, old top-level runtime files can be removed while preserving reusable cache directories:

```bash
find state -maxdepth 1 -type f ! -name '.gitkeep' -delete
```

If historical order-flow research outputs exist under `state/orderflow/`, they are not used by the live collector. The compact `aggtrades/` archive cache may be kept for reproducibility or removed independently.

## Kubernetes

Plain Kubernetes manifests (no Kustomize/Helm) live in `deploy/k8s/`.
Deployment instructions are in `deploy/k8s/README.md`; the normal entry point is:

```bash
./deploy/k8s/deploy.sh
```
