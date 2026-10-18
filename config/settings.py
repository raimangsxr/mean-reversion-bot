from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def _csv_strings(name: str, default: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in os.getenv(name, default).split(",") if v.strip())


def _csv_floats(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in os.getenv(name, default).split(",") if v.strip())


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# -----------------------------------------------------------------------------
# Binance Demo/live. V10.2 keeps the proven live execution path unchanged.
# -----------------------------------------------------------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
USE_TESTNET = _env_bool("USE_TESTNET", True)

SYMBOL = os.getenv("SYMBOL", "BTC/USDT:USDT")
LIVE_SYMBOLS = _csv_strings("LIVE_SYMBOLS", SYMBOL)
LIVE_MAX_CONCURRENT_POSITIONS = int(os.getenv("LIVE_MAX_CONCURRENT_POSITIONS", "3"))
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
MARKET_MODE = "future"
ENABLE_LONGS = _env_bool("ENABLE_LONGS", True)
ENABLE_SHORTS = _env_bool("ENABLE_SHORTS", False)
POSITION_MODE = "one_way"
MARGIN_MODE = os.getenv("MARGIN_MODE", "isolated").lower()
LEVERAGE = int(os.getenv("LEVERAGE", "2"))
TRIGGER_PRICE_TYPE = os.getenv("TRIGGER_PRICE_TYPE", "MARK_PRICE").upper()

# Frozen trigger/regime used by live and as the V10.2 research control.
LOOKBACK_WINDOW = int(os.getenv("LOOKBACK_WINDOW", "20"))
Z_THRESHOLD = float(os.getenv("Z_THRESHOLD", "2.2"))
TREND_WINDOW = int(os.getenv("TREND_WINDOW", "200"))
ATR_WINDOW = int(os.getenv("ATR_WINDOW", "14"))
REGIME_MODE = os.getenv("REGIME_MODE", "1h_sma200_4h_ema200").lower()
REGIME_TIMEFRAME = os.getenv("REGIME_TIMEFRAME", "4h")
REGIME_EMA_WINDOW = int(os.getenv("REGIME_EMA_WINDOW", "200"))
REGIME_SLOPE_LOOKBACK = int(os.getenv("REGIME_SLOPE_LOOKBACK", "6"))
ENTRY_MODE = os.getenv("ENTRY_MODE", "confirmed").lower()
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))

EXIT_MODE = os.getenv("EXIT_MODE", "fixed_rr").lower()
RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "1.8"))
MEAN_EXIT_Z = float(os.getenv("MEAN_EXIT_Z", "0.0"))
MAX_HOLDING_HOURS = int(os.getenv("MAX_HOLDING_HOURS", "24"))
BREAK_EVEN_TRIGGER_R = float(os.getenv("BREAK_EVEN_TRIGGER_R", "0.0"))
PARTIAL_TP_TRIGGER_R = float(os.getenv("PARTIAL_TP_TRIGGER_R", "0.0"))
PARTIAL_TP_FRACTION = float(os.getenv("PARTIAL_TP_FRACTION", "0.50"))
PARTIAL_TP_MOVE_STOP_TO_BE = _env_bool("PARTIAL_TP_MOVE_STOP_TO_BE", True)

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.005"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.30"))
CASH_BUFFER_PCT = float(os.getenv("CASH_BUFFER_PCT", "0.05"))
MAX_PORTFOLIO_RISK_PCT = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", "0.01"))

ENTRY_EXECUTION_MODE = os.getenv("ENTRY_EXECUTION_MODE", "pullback_limit").lower()
ENTRY_LIMIT_OFFSET_BPS = float(os.getenv("ENTRY_LIMIT_OFFSET_BPS", "3.0"))
ENTRY_LIMIT_WAIT_SECONDS = int(os.getenv("ENTRY_LIMIT_WAIT_SECONDS", "420"))
ENTRY_FALLBACK_CAP_BPS = float(os.getenv("ENTRY_FALLBACK_CAP_BPS", "5.0"))
ENTRY_PENDING_POLL_SECONDS = int(os.getenv("ENTRY_PENDING_POLL_SECONDS", "10"))
ENTRY_FALLBACK_MAX_LATENESS_SECONDS = int(
    os.getenv("ENTRY_FALLBACK_MAX_LATENESS_SECONDS", "60")
)

TAKE_PROFIT_ORDER_MODE = os.getenv("TAKE_PROFIT_ORDER_MODE", "maker_limit").lower()
STATE_FILE = os.getenv("STATE_FILE", "state/bot_state.json")
EXECUTION_JOURNAL_FILE = os.getenv("EXECUTION_JOURNAL_FILE", "state/execution_journal.jsonl")

CANDLE_CLOSE_DELAY_SECONDS = int(os.getenv("CANDLE_CLOSE_DELAY_SECONDS", "2"))
SIGNAL_MAX_AGE_SECONDS = int(os.getenv("SIGNAL_MAX_AGE_SECONDS", "60"))
ERROR_RETRY_SECONDS = int(os.getenv("ERROR_RETRY_SECONDS", "10"))
POSITION_POLL_SECONDS = int(os.getenv("POSITION_POLL_SECONDS", "30"))

# -----------------------------------------------------------------------------
# Shared research execution model.
# -----------------------------------------------------------------------------
BACKTEST_START = os.getenv("BACKTEST_START", "2020-01-01T00:00:00Z")
BACKTEST_END = os.getenv(
    "BACKTEST_END", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
)
BACKTEST_SYMBOLS = _csv_strings("BACKTEST_SYMBOLS", "BTC/USDT:USDT")
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
BACKTEST_FEE_STRESS_MULTIPLIERS = _csv_floats("BACKTEST_FEE_STRESS_MULTIPLIERS", "1.0,1.2")
BACKTEST_FUNDING_STRESS_MULTIPLIERS = _csv_floats(
    "BACKTEST_FUNDING_STRESS_MULTIPLIERS", "1.0,1.5,2.0"
)

# -----------------------------------------------------------------------------
# V10.2 research: z-score trigger + pre-registered confirmations.
# Thresholds are frozen from the completed frequency-matching experiment. No threshold tuning occurs in V10.2.
# 2022+ is DEVELOPMENT/diagnostic because shock1h was discovered by inspecting it;
# it must not be described as a fresh holdout.
# -----------------------------------------------------------------------------
V10_2_TAKER_CACHE_DIR = os.getenv("V10_2_TAKER_CACHE_DIR", "state/taker")
V10_2_RETURN_SHOCK_VOL_WINDOW = int(os.getenv("V10_2_RETURN_SHOCK_VOL_WINDOW", "100"))
V10_2_TAKER_LOOKBACK = int(os.getenv("V10_2_TAKER_LOOKBACK", "100"))
V10_2_TAKER_SELL_QUANTILE = float(os.getenv("V10_2_TAKER_SELL_QUANTILE", "0.90"))

V10_2_SHOCK1H_THRESHOLD = float(os.getenv("V10_2_SHOCK1H_THRESHOLD", "2.175"))
V10_2_SHOCK3H_THRESHOLD = float(os.getenv("V10_2_SHOCK3H_THRESHOLD", "1.725"))
V10_2_VWAP48H_THRESHOLD = float(os.getenv("V10_2_VWAP48H_THRESHOLD", "1.55"))
V10_2_TAKER_MIN_SHOCK_ATR = float(os.getenv("V10_2_TAKER_MIN_SHOCK_ATR", "0.925"))
V10_2_DEVELOPMENT_START_YEAR = int(os.getenv("V10_2_DEVELOPMENT_START_YEAR", "2022"))

# Sizing is studied only for the primary z-score + shock1h candidate.
V10_2_RISK_GRID = _csv_floats("V10_2_RISK_GRID", "0.005,0.0075,0.01,0.015")
V10_2_MAX_POSITION_GRID = _csv_floats("V10_2_MAX_POSITION_GRID", "0.30,0.50,0.70,1.00")
V10_2_SIZING_DD_LIMIT = float(os.getenv("V10_2_SIZING_DD_LIMIT", "0.10"))
V10_2_SIZING_MIN_PF = float(os.getenv("V10_2_SIZING_MIN_PF", "1.20"))

# Funding snapshot: historical rows are immutable after the first V10.2 snapshot.
V10_2_FUNDING_MAX_GAP_HOURS = float(os.getenv("V10_2_FUNDING_MAX_GAP_HOURS", "12"))
V10_2_CLEAN_OLD_ARTIFACTS = _env_bool("V10_2_CLEAN_OLD_ARTIFACTS", True)

V10_2_SUMMARY_FILE = os.getenv("V10_2_SUMMARY_FILE", "state/backtest_v10_2_summary.csv")
V10_2_ANNUAL_FILE = os.getenv("V10_2_ANNUAL_FILE", "state/backtest_v10_2_annual.csv")
V10_2_OPPORTUNITIES_FILE = os.getenv(
    "V10_2_OPPORTUNITIES_FILE", "state/backtest_v10_2_opportunities.csv"
)
V10_2_SIZING_FILE = os.getenv("V10_2_SIZING_FILE", "state/backtest_v10_2_sizing.csv")
V10_2_SENSITIVITY_FILE = os.getenv(
    "V10_2_SENSITIVITY_FILE", "state/backtest_v10_2_sensitivity.csv"
)
V10_2_STRESS_FILE = os.getenv("V10_2_STRESS_FILE", "state/backtest_v10_2_stress.csv")
V10_2_RECOMMENDATION_FILE = os.getenv(
    "V10_2_RECOMMENDATION_FILE", "state/backtest_v10_2_recommendation.json"
)


def validate_settings() -> None:
    if MARKET_MODE != "future":
        raise ValueError("El motor esta disenado para Binance USD-M Futures.")
    if POSITION_MODE != "one_way":
        raise ValueError("Solo se soporta POSITION_MODE='one_way'.")
    if MARGIN_MODE != "isolated":
        raise ValueError("El live requiere MARGIN_MODE=isolated.")
    if not 1 <= LEVERAGE <= 125:
        raise ValueError("LEVERAGE debe estar entre 1 y 125.")
    if TRIGGER_PRICE_TYPE not in {"MARK_PRICE", "CONTRACT_PRICE"}:
        raise ValueError("TRIGGER_PRICE_TYPE debe ser MARK_PRICE o CONTRACT_PRICE.")
    if TAKE_PROFIT_ORDER_MODE != "maker_limit":
        raise ValueError("V10.2 requiere TAKE_PROFIT_ORDER_MODE=maker_limit.")
    if ENTRY_EXECUTION_MODE != "pullback_limit":
        raise ValueError("V10.2 mantiene ENTRY_EXECUTION_MODE=pullback_limit.")
    if ENTRY_MODE != "confirmed":
        raise ValueError("V10.2 mantiene ENTRY_MODE=confirmed.")
    if REGIME_MODE != "1h_sma200_4h_ema200":
        raise ValueError("V10.2 mantiene REGIME_MODE=1h_sma200_4h_ema200.")
    if ENTRY_LIMIT_OFFSET_BPS <= 0 or ENTRY_LIMIT_WAIT_SECONDS <= 0:
        raise ValueError("Offset/wait de entrada deben ser positivos.")
    if ENTRY_LIMIT_WAIT_SECONDS % 60 != 0:
        raise ValueError("ENTRY_LIMIT_WAIT_SECONDS debe ser multiplo de 60 para research 1m.")
    if ENTRY_FALLBACK_CAP_BPS < 0 or ENTRY_FALLBACK_MAX_LATENESS_SECONDS < 0:
        raise ValueError("Cap/lateness no pueden ser negativos.")
    if ENTRY_PENDING_POLL_SECONDS <= 0 or POSITION_POLL_SECONDS <= 0:
        raise ValueError("Los intervalos de polling deben ser positivos.")
    if CANDLE_CLOSE_DELAY_SECONDS < 0 or SIGNAL_MAX_AGE_SECONDS <= 0 or ERROR_RETRY_SECONDS <= 0:
        raise ValueError("Configuracion de scheduler invalida.")
    if not 0 < RISK_PER_TRADE <= 1 or not 0 < MAX_POSITION_PCT <= 1:
        raise ValueError("RISK_PER_TRADE/MAX_POSITION_PCT deben estar en (0,1].")
    if not 0 <= CASH_BUFFER_PCT < 1 or not 0 < MAX_PORTFOLIO_RISK_PCT <= 1:
        raise ValueError("Buffers/riesgo de portfolio invalidos.")
    if not LIVE_SYMBOLS or LIVE_MAX_CONCURRENT_POSITIONS < 1:
        raise ValueError("Configuracion live de simbolos/concurrencia invalida.")
    if ATR_MULTIPLIER <= 0 or RISK_REWARD_RATIO <= 0:
        raise ValueError("ATR_MULTIPLIER/RISK_REWARD_RATIO deben ser positivos.")
    if MEAN_EXIT_Z < 0 or MAX_HOLDING_HOURS <= 0:
        raise ValueError("MEAN_EXIT_Z/MAX_HOLDING_HOURS invalidos.")
    if BREAK_EVEN_TRIGGER_R < 0 or PARTIAL_TP_TRIGGER_R < 0:
        raise ValueError("Triggers de salida no pueden ser negativos.")
    if not 0 < PARTIAL_TP_FRACTION < 1:
        raise ValueError("PARTIAL_TP_FRACTION debe estar en (0,1).")
    if BACKTEST_TAKER_FEE < 0 or BACKTEST_MAKER_FEE < 0:
        raise ValueError("Fees de backtest no pueden ser negativas.")
    if len(BACKTEST_SYMBOLS) != 1:
        raise ValueError("V10.2 research requiere exactamente un simbolo.")
    if not BACKTEST_MARKET_SLIPPAGE_BPS or any(v < 0 for v in BACKTEST_MARKET_SLIPPAGE_BPS):
        raise ValueError("BACKTEST_MARKET_SLIPPAGE_BPS debe contener valores >= 0.")
    if BACKTEST_TP_CROSS_TICKS < 1:
        raise ValueError("BACKTEST_TP_CROSS_TICKS debe ser >= 1.")
    if not 0 < BACKTEST_MAX_PORTFOLIO_RISK_PCT <= 1 or BACKTEST_MAX_CONCURRENT_POSITIONS < 1:
        raise ValueError("Configuracion de portfolio de backtest invalida.")
    if not 0 < BACKTEST_DD_LIMIT < 1 or not 0 < V10_2_SIZING_DD_LIMIT < 1:
        raise ValueError("Limites de DD deben estar en (0,1).")
    if V10_2_SIZING_MIN_PF < 1.0:
        raise ValueError("V10_2_SIZING_MIN_PF debe ser >= 1.0.")
    if BACKTEST_MIN_TRAIN_TRADES < 1 or BACKTEST_FIRST_TEST_YEAR < 2000:
        raise ValueError("Configuracion temporal de research invalida.")
    if not 0 < BACKTEST_PARTIAL_FRACTION < 1:
        raise ValueError("BACKTEST_PARTIAL_FRACTION debe estar en (0,1).")
    if any(v <= 0 for v in BACKTEST_FEE_STRESS_MULTIPLIERS + BACKTEST_FUNDING_STRESS_MULTIPLIERS):
        raise ValueError("Multiplicadores de stress deben ser > 0.")
    if any(not 0 < v <= 1 for v in V10_2_RISK_GRID + V10_2_MAX_POSITION_GRID):
        raise ValueError("Grids V10.2 de risk/max-position deben estar en (0,1].")
    if V10_2_FUNDING_MAX_GAP_HOURS <= 8:
        raise ValueError("V10_2_FUNDING_MAX_GAP_HOURS debe ser > 8h.")
