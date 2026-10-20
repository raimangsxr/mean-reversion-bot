from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def _csv_floats(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in os.getenv(name, default).split(",") if v.strip())


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# -----------------------------------------------------------------------------
# Binance USD-M Demo execution.
# -----------------------------------------------------------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
USE_TESTNET = _env_bool("USE_TESTNET", True)

TIMEFRAME = os.getenv("TIMEFRAME", "1h")
MARKET_MODE = "future"
POSITION_MODE = "one_way"
MARGIN_MODE = os.getenv("MARGIN_MODE", "isolated").lower()
LEVERAGE = int(os.getenv("LEVERAGE", "2"))
TRIGGER_PRICE_TYPE = os.getenv("TRIGGER_PRICE_TYPE", "MARK_PRICE").upper()

# Frozen signal/regime.
LOOKBACK_WINDOW = int(os.getenv("LOOKBACK_WINDOW", "20"))
Z_THRESHOLD = float(os.getenv("Z_THRESHOLD", "2.2"))
SHOCK1H_THRESHOLD = float(os.getenv("SHOCK1H_THRESHOLD", "2.175"))
RETURN_SHOCK_VOL_WINDOW = int(os.getenv("RETURN_SHOCK_VOL_WINDOW", "100"))
TREND_WINDOW = int(os.getenv("TREND_WINDOW", "200"))
ATR_WINDOW = int(os.getenv("ATR_WINDOW", "14"))
REGIME_MODE = os.getenv("REGIME_MODE", "1h_sma200_4h_ema200").lower()
REGIME_TIMEFRAME = os.getenv("REGIME_TIMEFRAME", "4h")
REGIME_EMA_WINDOW = int(os.getenv("REGIME_EMA_WINDOW", "200"))
REGIME_SLOPE_LOOKBACK = int(os.getenv("REGIME_SLOPE_LOOKBACK", "6"))
ENTRY_MODE = os.getenv("ENTRY_MODE", "confirmed").lower()
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))

# Frozen exit/execution model.
EXIT_MODE = os.getenv("EXIT_MODE", "fixed_rr").lower()
RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "1.8"))
MEAN_EXIT_Z = float(os.getenv("MEAN_EXIT_Z", "0.0"))
MAX_HOLDING_HOURS = int(os.getenv("MAX_HOLDING_HOURS", "24"))
BREAK_EVEN_TRIGGER_R = float(os.getenv("BREAK_EVEN_TRIGGER_R", "0.0"))
PARTIAL_TP_TRIGGER_R = float(os.getenv("PARTIAL_TP_TRIGGER_R", "0.0"))
PARTIAL_TP_FRACTION = float(os.getenv("PARTIAL_TP_FRACTION", "0.50"))
PARTIAL_TP_MOVE_STOP_TO_BE = _env_bool("PARTIAL_TP_MOVE_STOP_TO_BE", True)

ENTRY_EXECUTION_MODE = os.getenv("ENTRY_EXECUTION_MODE", "pullback_limit").lower()
ENTRY_LIMIT_OFFSET_BPS = float(os.getenv("ENTRY_LIMIT_OFFSET_BPS", "3.0"))
ENTRY_LIMIT_WAIT_SECONDS = int(os.getenv("ENTRY_LIMIT_WAIT_SECONDS", "420"))
ENTRY_FALLBACK_CAP_BPS = float(os.getenv("ENTRY_FALLBACK_CAP_BPS", "5.0"))
ENTRY_PENDING_POLL_SECONDS = int(os.getenv("ENTRY_PENDING_POLL_SECONDS", "10"))
ENTRY_FALLBACK_MAX_LATENESS_SECONDS = int(
    os.getenv("ENTRY_FALLBACK_MAX_LATENESS_SECONDS", "60")
)
TAKE_PROFIT_ORDER_MODE = os.getenv("TAKE_PROFIT_ORDER_MODE", "maker_limit").lower()

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
CASH_BUFFER_PCT = float(os.getenv("CASH_BUFFER_PCT", "0.05"))

# Runtime files. STATE_FILE is a base name; main.py derives one state file per symbol.
STATE_FILE = os.getenv("STATE_FILE", "state/bot_state.json")
EXECUTION_JOURNAL_FILE = os.getenv("EXECUTION_JOURNAL_FILE", "state/execution_journal.jsonl")
FORWARD_JOURNAL_FILE = os.getenv("FORWARD_JOURNAL_FILE", "state/forward.jsonl")

CANDLE_CLOSE_DELAY_SECONDS = int(os.getenv("CANDLE_CLOSE_DELAY_SECONDS", "2"))
SIGNAL_MAX_AGE_SECONDS = int(os.getenv("SIGNAL_MAX_AGE_SECONDS", "60"))
ERROR_RETRY_SECONDS = int(os.getenv("ERROR_RETRY_SECONDS", "10"))
POSITION_POLL_SECONDS = int(os.getenv("POSITION_POLL_SECONDS", "30"))
BUILD_SHA = os.getenv("BUILD_SHA", "").strip()

# -----------------------------------------------------------------------------
# Active V12 forward contract. Components and sizing are code-frozen.
# V12_START_UTC deliberately has a versioned environment key so an old generic
# forward boundary cannot be reused accidentally when starting this clean run.
# -----------------------------------------------------------------------------
STRATEGY_VERSION = "v12-forward"
COMPONENT_SPECS = (
    ("BTC/USDT:USDT", "long", 0.0075),
    ("BNB/USDT:USDT", "long", 0.0050),
    ("DOT/USDT:USDT", "short", 0.0075),
    ("ADA/USDT:USDT", "short", 0.0050),
)
SYMBOLS = tuple(spec[0] for spec in COMPONENT_SPECS)
SIDE_BY_SYMBOL = {spec[0]: spec[1] for spec in COMPONENT_SPECS}
RISK_BY_SYMBOL = {spec[0]: spec[2] for spec in COMPONENT_SPECS}
MAX_POSITION_PCT = 0.50
MAX_PORTFOLIO_RISK_PCT = 0.015
MAX_CONCURRENT_POSITIONS = 3

# Frozen causal executed-flow gate. These are code constants, not tunable env vars.
ORDERFLOW_DECISION_SECONDS = 60
ORDERFLOW_IMBALANCE_THRESHOLD = 0.20
ORDERFLOW_ADVERSE_ACCEPTANCE_BPS = 5.0
ORDERFLOW_ABSORPTION_TOLERANCE_BPS = 2.0
ORDERFLOW_TOTAL_ENTRY_DEADLINE_SECONDS = 420
# Operational grace only: if complete WebSocket coverage cannot be proven shortly
# after t+60s, the signal fails closed rather than trading on partial flow.
ORDERFLOW_COVERAGE_GRACE_SECONDS = 10

FORWARD_ONLY_DEMO = _env_bool("FORWARD_ONLY_DEMO", True)
FORWARD_EXPERIMENT_ID = os.getenv("FORWARD_EXPERIMENT_ID", "v12").strip()
FORWARD_START_UTC = os.getenv("V12_START_UTC", "").strip()


def frozen_config() -> dict[str, object]:
    return {
        "strategy_version": STRATEGY_VERSION,
        "components": [
            {"symbol": symbol, "side": side, "risk_per_trade": risk}
            for symbol, side, risk in COMPONENT_SPECS
        ],
        "timeframe": TIMEFRAME,
        "lookback_window": LOOKBACK_WINDOW,
        "z_threshold": Z_THRESHOLD,
        "shock1h_threshold": SHOCK1H_THRESHOLD,
        "return_shock_vol_window": RETURN_SHOCK_VOL_WINDOW,
        "trend_window": TREND_WINDOW,
        "atr_window": ATR_WINDOW,
        "regime_mode": REGIME_MODE,
        "regime_timeframe": REGIME_TIMEFRAME,
        "regime_ema_window": REGIME_EMA_WINDOW,
        "entry_mode": ENTRY_MODE,
        "atr_multiplier": ATR_MULTIPLIER,
        "entry_execution_mode": ENTRY_EXECUTION_MODE,
        "entry_limit_offset_bps": ENTRY_LIMIT_OFFSET_BPS,
        "entry_limit_wait_seconds": ENTRY_LIMIT_WAIT_SECONDS,
        "entry_fallback_cap_bps": ENTRY_FALLBACK_CAP_BPS,
        "exit_mode": EXIT_MODE,
        "risk_reward_ratio": RISK_REWARD_RATIO,
        "take_profit_order_mode": TAKE_PROFIT_ORDER_MODE,
        "max_position_pct": MAX_POSITION_PCT,
        "max_portfolio_risk_pct": MAX_PORTFOLIO_RISK_PCT,
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
        "orderflow_decision_seconds": ORDERFLOW_DECISION_SECONDS,
        "orderflow_imbalance_threshold": ORDERFLOW_IMBALANCE_THRESHOLD,
        "orderflow_adverse_acceptance_bps": ORDERFLOW_ADVERSE_ACCEPTANCE_BPS,
        "orderflow_total_entry_deadline_seconds": ORDERFLOW_TOTAL_ENTRY_DEADLINE_SECONDS,
        "orderflow_coverage_grace_seconds": ORDERFLOW_COVERAGE_GRACE_SECONDS,
        "break_even_trigger_r": BREAK_EVEN_TRIGGER_R,
        "partial_tp_trigger_r": PARTIAL_TP_TRIGGER_R,
        "forward_only_demo": FORWARD_ONLY_DEMO,
        "use_testnet": USE_TESTNET,
    }


# -----------------------------------------------------------------------------
# Replay infrastructure used by forward_report.py and generic research tests.
# Caches are intentionally retained across cleanups because they are immutable/
# append-only market data, not experiment output.
# -----------------------------------------------------------------------------
BACKTEST_START = os.getenv("BACKTEST_START", "2020-01-01T00:00:00Z")
BACKTEST_END = os.getenv(
    "BACKTEST_END", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
)
BACKTEST_TAKER_FEE = float(os.getenv("BACKTEST_TAKER_FEE", "0.0005"))
BACKTEST_MAKER_FEE = float(os.getenv("BACKTEST_MAKER_FEE", "0.0002"))
BACKTEST_MARKET_SLIPPAGE_BPS = _csv_floats("BACKTEST_MARKET_SLIPPAGE_BPS", "1,2,3,5")
BACKTEST_PRIMARY_SLIPPAGE_BPS = float(os.getenv("BACKTEST_PRIMARY_SLIPPAGE_BPS", "2"))
BACKTEST_TP_CROSS_TICKS = int(os.getenv("BACKTEST_TP_CROSS_TICKS", "1"))
BACKTEST_REQUIRE_FUNDING = _env_bool("BACKTEST_REQUIRE_FUNDING", True)
BACKTEST_FUNDING_CACHE_DIR = os.getenv("BACKTEST_FUNDING_CACHE_DIR", "state/funding")
BACKTEST_MINUTE_CACHE_DIR = os.getenv("BACKTEST_MINUTE_CACHE_DIR", "state/minute")
BACKTEST_HISTORY_CACHE_DIR = os.getenv("BACKTEST_HISTORY_CACHE_DIR", "state/history")
BACKTEST_MAX_PORTFOLIO_RISK_PCT = float(os.getenv("BACKTEST_MAX_PORTFOLIO_RISK_PCT", "0.015"))
BACKTEST_MAX_CONCURRENT_POSITIONS = int(os.getenv("BACKTEST_MAX_CONCURRENT_POSITIONS", "3"))
BACKTEST_DD_LIMIT = float(os.getenv("BACKTEST_DD_LIMIT", "0.10"))
BACKTEST_MIN_TRAIN_TRADES = int(os.getenv("BACKTEST_MIN_TRAIN_TRADES", "60"))
BACKTEST_FIRST_TEST_YEAR = int(os.getenv("BACKTEST_FIRST_TEST_YEAR", "2022"))
BACKTEST_PARTIAL_FRACTION = float(os.getenv("BACKTEST_PARTIAL_FRACTION", "0.50"))
FUNDING_MAX_GAP_HOURS = float(os.getenv("FUNDING_MAX_GAP_HOURS", "12"))

# Optional generic signal-research defaults retained by src/research/signals.py.
RESEARCH_RETURN_SHOCK_VOL_WINDOW = int(os.getenv("RESEARCH_RETURN_SHOCK_VOL_WINDOW", "100"))
RESEARCH_TAKER_LOOKBACK = int(os.getenv("RESEARCH_TAKER_LOOKBACK", "100"))
RESEARCH_TAKER_SELL_QUANTILE = float(os.getenv("RESEARCH_TAKER_SELL_QUANTILE", "0.90"))
RESEARCH_TAKER_MIN_SHOCK_ATR = float(os.getenv("RESEARCH_TAKER_MIN_SHOCK_ATR", "0.925"))


def _validate_base_settings() -> None:
    if MARKET_MODE != "future":
        raise ValueError("The engine is designed for Binance USD-M Futures.")
    if POSITION_MODE != "one_way":
        raise ValueError("Only POSITION_MODE='one_way' is supported.")
    if MARGIN_MODE != "isolated":
        raise ValueError("Forward execution requires MARGIN_MODE=isolated.")
    if not 1 <= LEVERAGE <= 125:
        raise ValueError("LEVERAGE must be between 1 and 125.")
    if TRIGGER_PRICE_TYPE not in {"MARK_PRICE", "CONTRACT_PRICE"}:
        raise ValueError("TRIGGER_PRICE_TYPE must be MARK_PRICE or CONTRACT_PRICE.")
    if ENTRY_LIMIT_OFFSET_BPS <= 0 or ENTRY_LIMIT_WAIT_SECONDS <= 0:
        raise ValueError("Entry offset/wait must be positive.")
    if ENTRY_LIMIT_WAIT_SECONDS % 60 != 0:
        raise ValueError("ENTRY_LIMIT_WAIT_SECONDS must be a multiple of 60.")
    if ENTRY_FALLBACK_CAP_BPS < 0 or ENTRY_FALLBACK_MAX_LATENESS_SECONDS < 0:
        raise ValueError("Fallback cap/lateness cannot be negative.")
    if ENTRY_PENDING_POLL_SECONDS <= 0 or POSITION_POLL_SECONDS <= 0:
        raise ValueError("Polling intervals must be positive.")
    if CANDLE_CLOSE_DELAY_SECONDS < 0 or SIGNAL_MAX_AGE_SECONDS <= 0 or ERROR_RETRY_SECONDS <= 0:
        raise ValueError("Invalid scheduler configuration.")
    if not 0 <= CASH_BUFFER_PCT < 1:
        raise ValueError("CASH_BUFFER_PCT must be in [0,1).")
    if ATR_MULTIPLIER <= 0 or RISK_REWARD_RATIO <= 0:
        raise ValueError("ATR_MULTIPLIER/RISK_REWARD_RATIO must be positive.")
    if MEAN_EXIT_Z < 0 or MAX_HOLDING_HOURS <= 0:
        raise ValueError("Invalid mean/time exit configuration.")
    if BREAK_EVEN_TRIGGER_R < 0 or PARTIAL_TP_TRIGGER_R < 0:
        raise ValueError("Exit triggers cannot be negative.")
    if not 0 < PARTIAL_TP_FRACTION < 1:
        raise ValueError("PARTIAL_TP_FRACTION must be in (0,1).")
    if BACKTEST_TAKER_FEE < 0 or BACKTEST_MAKER_FEE < 0:
        raise ValueError("Replay fees cannot be negative.")
    if not BACKTEST_MARKET_SLIPPAGE_BPS or any(v < 0 for v in BACKTEST_MARKET_SLIPPAGE_BPS):
        raise ValueError("BACKTEST_MARKET_SLIPPAGE_BPS must contain values >= 0.")
    if BACKTEST_TP_CROSS_TICKS < 1:
        raise ValueError("BACKTEST_TP_CROSS_TICKS must be >= 1.")
    if not 0 < BACKTEST_MAX_PORTFOLIO_RISK_PCT <= 1 or BACKTEST_MAX_CONCURRENT_POSITIONS < 1:
        raise ValueError("Invalid replay portfolio limits.")
    if not 0 < BACKTEST_DD_LIMIT < 1:
        raise ValueError("BACKTEST_DD_LIMIT must be in (0,1).")
    if BACKTEST_MIN_TRAIN_TRADES < 1 or BACKTEST_FIRST_TEST_YEAR < 2000:
        raise ValueError("Invalid generic research temporal settings.")
    if FUNDING_MAX_GAP_HOURS <= 8:
        raise ValueError("FUNDING_MAX_GAP_HOURS must be > 8h.")


def validate_forward_settings() -> None:
    """Validate the single active V12 forward contract and fail closed on drift."""
    _validate_base_settings()

    expected_specs = (
        ("BTC/USDT:USDT", "long", 0.0075),
        ("BNB/USDT:USDT", "long", 0.0050),
        ("DOT/USDT:USDT", "short", 0.0075),
        ("ADA/USDT:USDT", "short", 0.0050),
    )
    if COMPONENT_SPECS != expected_specs:
        raise ValueError(f"V12 components/sizing are frozen at {expected_specs}.")
    if MAX_POSITION_PCT != 0.50:
        raise ValueError("V12 max position is frozen at 50%.")
    if MAX_PORTFOLIO_RISK_PCT != 0.015:
        raise ValueError("V12 portfolio risk cap is frozen at 1.5%.")
    if MAX_CONCURRENT_POSITIONS != 3:
        raise ValueError("V12 max concurrency is frozen at 3.")
    if ORDERFLOW_DECISION_SECONDS != 60:
        raise ValueError("V12 order-flow decision delay is frozen at 60s.")
    if ORDERFLOW_IMBALANCE_THRESHOLD != 0.20:
        raise ValueError("V12 order-flow imbalance threshold is frozen at 20%.")
    if ORDERFLOW_ADVERSE_ACCEPTANCE_BPS != 5.0:
        raise ValueError("V12 adverse-acceptance threshold is frozen at 5bp.")
    if ORDERFLOW_TOTAL_ENTRY_DEADLINE_SECONDS != 420:
        raise ValueError("V12 total entry deadline is frozen at 420s.")
    if ORDERFLOW_TOTAL_ENTRY_DEADLINE_SECONDS != ENTRY_LIMIT_WAIT_SECONDS:
        raise ValueError("Order-flow total deadline must equal ENTRY_LIMIT_WAIT_SECONDS.")
    if ORDERFLOW_COVERAGE_GRACE_SECONDS < 0:
        raise ValueError("ORDERFLOW_COVERAGE_GRACE_SECONDS cannot be negative.")
    if not FORWARD_ONLY_DEMO:
        raise ValueError("V12 requires FORWARD_ONLY_DEMO=True.")
    if not USE_TESTNET:
        raise ValueError("V12 execution is Demo-only; USE_TESTNET must be True.")
    if not FORWARD_EXPERIMENT_ID:
        raise ValueError("FORWARD_EXPERIMENT_ID cannot be empty.")
    if not FORWARD_START_UTC:
        raise ValueError("Set V12_START_UTC before the first V12 run and never move it afterwards.")
    try:
        start = datetime.fromisoformat(FORWARD_START_UTC.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("V12_START_UTC must be ISO-8601 UTC.") from exc
    if start.tzinfo is None or start.utcoffset() != timezone.utc.utcoffset(start):
        raise ValueError("V12_START_UTC must include UTC timezone information.")

    frozen = {
        "TIMEFRAME": (TIMEFRAME, "1h"),
        "LOOKBACK_WINDOW": (LOOKBACK_WINDOW, 20),
        "Z_THRESHOLD": (Z_THRESHOLD, 2.2),
        "SHOCK1H_THRESHOLD": (SHOCK1H_THRESHOLD, 2.175),
        "RETURN_SHOCK_VOL_WINDOW": (RETURN_SHOCK_VOL_WINDOW, 100),
        "TREND_WINDOW": (TREND_WINDOW, 200),
        "ATR_WINDOW": (ATR_WINDOW, 14),
        "REGIME_MODE": (REGIME_MODE, "1h_sma200_4h_ema200"),
        "REGIME_TIMEFRAME": (REGIME_TIMEFRAME, "4h"),
        "REGIME_EMA_WINDOW": (REGIME_EMA_WINDOW, 200),
        "ENTRY_MODE": (ENTRY_MODE, "confirmed"),
        "ATR_MULTIPLIER": (ATR_MULTIPLIER, 1.5),
        "ENTRY_EXECUTION_MODE": (ENTRY_EXECUTION_MODE, "pullback_limit"),
        "ENTRY_LIMIT_OFFSET_BPS": (ENTRY_LIMIT_OFFSET_BPS, 3.0),
        "ENTRY_LIMIT_WAIT_SECONDS": (ENTRY_LIMIT_WAIT_SECONDS, 420),
        "ENTRY_FALLBACK_CAP_BPS": (ENTRY_FALLBACK_CAP_BPS, 5.0),
        "EXIT_MODE": (EXIT_MODE, "fixed_rr"),
        "RISK_REWARD_RATIO": (RISK_REWARD_RATIO, 1.8),
        "TAKE_PROFIT_ORDER_MODE": (TAKE_PROFIT_ORDER_MODE, "maker_limit"),
        "BREAK_EVEN_TRIGGER_R": (BREAK_EVEN_TRIGGER_R, 0.0),
        "PARTIAL_TP_TRIGGER_R": (PARTIAL_TP_TRIGGER_R, 0.0),
    }
    drift = [name for name, (actual, expected) in frozen.items() if actual != expected]
    if drift:
        details = ", ".join(
            f"{name}={frozen[name][0]!r} expected {frozen[name][1]!r}" for name in drift
        )
        raise ValueError(f"V12 frozen configuration drift: {details}")
