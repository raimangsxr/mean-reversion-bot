V7 live execution changes
=========================

Strategy adopted for live:
- LONG only
- confirmed entry
- local regime: close 1h > SMA200 1h
- HTF regime: close 4h > EMA200 4h
- ATR stop = 1.5
- RR = 1.5

Execution:
- entry: MARKET/taker
- stop: STOP_MARKET reduceOnly
- take profit: LIMIT reduceOnly postOnly (maker intent)
- fallback compatibility: TAKE_PROFIT_ORDER_MODE=trigger_market

Telemetry:
- append-only JSONL at state/execution_journal.jsonl by default
- entry orderbook snapshot and slippage
- fill/fee/trade enrichment when available
- TP partial fills
- detected TP/SL closures
- market/emergency exits

Important: apply while the bot is flat. Existing V6/V5 trigger TP state is migrated conservatively,
but changing execution policy with an open position is intentionally discouraged.
