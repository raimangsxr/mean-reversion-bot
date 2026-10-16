import pandas as pd

from backtest import EntryVariant, resolve_entry


def minute_frame(rows):
    idx = pd.date_range("2026-01-01T01:00:00Z", periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=idx)


def test_v8_final_limit_is_anchored_to_signal_close_not_t0_open():
    # signal close=100 -> target=99.97. t0 open=100.20 only anchors no-chase.
    df = minute_frame([
        {"open": 100.20, "high": 100.30, "low": 99.96, "close": 100.10, "volume": 1},
        {"open": 100.10, "high": 100.20, "low": 100.00, "close": 100.10, "volume": 1},
    ])
    variant = EntryVariant(
        "pullback_limit", "conditional", limit_offset_bps=3.0,
        wait_minutes=1, max_chase_bps=5.0,
    )

    fill = resolve_entry(variant, df, signal_close=100.0, market_slippage_bps=2.0)

    assert fill is not None
    assert fill.liquidity == "maker"
    assert fill.fill_price == 99.97
    assert fill.baseline_open == 100.20


def test_v8_final_no_chase_cap_is_anchored_to_t0_open():
    rows = []
    for i in range(8):
        open_price = 100.0 if i == 0 else 100.04
        rows.append({
            "open": open_price,
            "high": open_price + 0.02,
            "low": 99.98,  # never crosses LIMIT 99.97
            "close": open_price,
            "volume": 1,
        })
    df = minute_frame(rows)
    variant = EntryVariant(
        "pullback_limit", "conditional", limit_offset_bps=3.0,
        wait_minutes=7, max_chase_bps=5.0,
    )

    fill = resolve_entry(variant, df, signal_close=100.0, market_slippage_bps=2.0)

    assert fill is not None
    assert fill.route == "market_fallback_conditional"
    assert fill.raw_price == 100.04  # <= t0 100.00 + 5bp (=100.05)


def test_v8_final_marketable_at_t0_does_not_claim_maker_but_can_fallback():
    rows = []
    for i in range(8):
        # target from signal close 100 is 99.97; t0 opens below it, so a BUY
        # postOnly would be marketable and must not be counted as maker.
        open_price = 99.95 if i == 0 else 99.97
        rows.append({
            "open": open_price,
            "high": 100.00,
            "low": 99.90,
            "close": open_price,
            "volume": 1,
        })
    df = minute_frame(rows)
    variant = EntryVariant(
        "pullback_limit", "conditional", limit_offset_bps=3.0,
        wait_minutes=7, max_chase_bps=5.0,
    )

    fill = resolve_entry(variant, df, signal_close=100.0, market_slippage_bps=0.0)

    assert fill is not None
    assert fill.liquidity == "taker"
    assert fill.route == "market_fallback_conditional"
