from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def _csv_strings(name: str, default: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in os.getenv(name, default).split(",") if v.strip())


def _csv_floats(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in os.getenv(name, default).split(",") if v.strip())


def _csv_ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in os.getenv(name, default).split(",") if v.strip())


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
USE_TESTNET = _env_bool("USE_TESTNET", True)

# -----------------------------------------------------------------------------
# Mercado live. V9 research es multi-simbolo, pero el live permanece BTC-only
# por defecto hasta que el walk-forward/cross-sectional cierre la seleccion.
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Estrategia base congelada para V9.
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Live V9.2: candidato BTC LONG validado en V9.1.
# -----------------------------------------------------------------------------
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

# Entrada congelada V9: pullback + no-chase.
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
EXECUTION_JOURNAL_FILE = os.getenv(
    "EXECUTION_JOURNAL_FILE", "state/execution_journal.jsonl"
)

# Scheduler live: se alinea al cierre de la vela en vez de dormir 60 s a ciegas.
CANDLE_CLOSE_DELAY_SECONDS = int(os.getenv("CANDLE_CLOSE_DELAY_SECONDS", "2"))
SIGNAL_MAX_AGE_SECONDS = int(os.getenv("SIGNAL_MAX_AGE_SECONDS", "60"))
ERROR_RETRY_SECONDS = int(os.getenv("ERROR_RETRY_SECONDS", "10"))
POSITION_POLL_SECONDS = int(os.getenv("POSITION_POLL_SECONDS", "30"))
ENTRY_PENDING_POLL_SECONDS = int(os.getenv("ENTRY_PENDING_POLL_SECONDS", "10"))

# -----------------------------------------------------------------------------
# V9 research: costes y realismo de ejecucion.
# -----------------------------------------------------------------------------
BACKTEST_START = os.getenv("BACKTEST_START", "2020-01-01T00:00:00Z")
BACKTEST_END = os.getenv(
    "BACKTEST_END",
    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
)
BACKTEST_SYMBOLS = _csv_strings(
    "BACKTEST_SYMBOLS",
    "BTC/USDT:USDT",
)
# V9.2 optimiza exclusivamente BTC. Otros simbolos pueden anadirse manualmente
# para experimentos separados, pero no forman parte del flujo principal.
BACKTEST_SELECTION_SYMBOLS = _csv_strings(
    "BACKTEST_SELECTION_SYMBOLS",
    "BTC/USDT:USDT",
)
BACKTEST_TAKER_FEE = float(os.getenv("BACKTEST_TAKER_FEE", "0.0005"))
BACKTEST_MAKER_FEE = float(os.getenv("BACKTEST_MAKER_FEE", "0.0002"))
BACKTEST_MARKET_SLIPPAGE_BPS = _csv_floats(
    "BACKTEST_MARKET_SLIPPAGE_BPS", "1,2,3,5"
)
BACKTEST_PRIMARY_SLIPPAGE_BPS = float(
    os.getenv("BACKTEST_PRIMARY_SLIPPAGE_BPS", "2")
)
BACKTEST_TP_CROSS_TICKS = int(os.getenv("BACKTEST_TP_CROSS_TICKS", "1"))
BACKTEST_REQUIRE_FUNDING = _env_bool("BACKTEST_REQUIRE_FUNDING", True)
BACKTEST_FUNDING_CACHE_DIR = os.getenv(
    "BACKTEST_FUNDING_CACHE_DIR", "state/funding"
)
BACKTEST_MINUTE_CACHE_DIR = os.getenv(
    "BACKTEST_MINUTE_CACHE_DIR", "state/minute"
)
BACKTEST_HISTORY_CACHE_DIR = os.getenv(
    "BACKTEST_HISTORY_CACHE_DIR", "state/history"
)

# Salidas V9. El parser acepta perfiles compactos definidos en src/research/v9.py.
BACKTEST_EXIT_PROFILES = _csv_strings(
    "BACKTEST_EXIT_PROFILES",
    "fixed_rr",
)
BACKTEST_TP_RATIOS = _csv_floats("BACKTEST_TP_RATIOS", "1.5,1.8")
BACKTEST_MAX_POSITION_PCT_GRID = _csv_floats(
    "BACKTEST_MAX_POSITION_PCT_GRID", "0.10,0.30,0.50,0.60"
)
BACKTEST_RISK_PER_TRADE_GRID = _csv_floats(
    "BACKTEST_RISK_PER_TRADE_GRID", "0.005"
)
BACKTEST_SIDE_MODES = _csv_strings(
    "BACKTEST_SIDE_MODES", "long_only"
)
BACKTEST_MAX_PORTFOLIO_RISK_PCT = float(
    os.getenv("BACKTEST_MAX_PORTFOLIO_RISK_PCT", "0.015")
)
BACKTEST_MAX_CONCURRENT_POSITIONS = int(
    os.getenv("BACKTEST_MAX_CONCURRENT_POSITIONS", "3")
)
BACKTEST_DD_LIMIT = float(os.getenv("BACKTEST_DD_LIMIT", "0.10"))
# V9.2: robustez no binaria. Un ano practicamente plano no invalida por si
# solo una configuracion que multiplica el beneficio con DD contenido.
BACKTEST_MIN_POSITIVE_YEARS = int(os.getenv("BACKTEST_MIN_POSITIVE_YEARS", "4"))
BACKTEST_MIN_WORST_YEAR_PF = float(os.getenv("BACKTEST_MIN_WORST_YEAR_PF", "0.90"))
BACKTEST_MIN_WORST_YEAR_RETURN = float(os.getenv("BACKTEST_MIN_WORST_YEAR_RETURN", "-0.01"))
BACKTEST_MIN_GLOBAL_PF = float(os.getenv("BACKTEST_MIN_GLOBAL_PF", "1.20"))
BACKTEST_MIN_TRAIN_TRADES = int(os.getenv("BACKTEST_MIN_TRAIN_TRADES", "60"))
BACKTEST_FIRST_TEST_YEAR = int(os.getenv("BACKTEST_FIRST_TEST_YEAR", "2022"))
BACKTEST_PARTIAL_FRACTION = float(os.getenv("BACKTEST_PARTIAL_FRACTION", "0.50"))

# Rendimiento del research. La seleccion completa se hace solo al slippage
# primario; los otros slippages se recalculan unicamente para finalistas.
BACKTEST_WORKERS = int(os.getenv("BACKTEST_WORKERS", "0"))  # 0 = auto
BACKTEST_REUSE_OPPORTUNITIES = _env_bool("BACKTEST_REUSE_OPPORTUNITIES", True)
BACKTEST_REBUILD_OPPORTUNITIES = _env_bool("BACKTEST_REBUILD_OPPORTUNITIES", False)
BACKTEST_SENSITIVITY_TOP_N = int(os.getenv("BACKTEST_SENSITIVITY_TOP_N", "12"))
BACKTEST_FEE_STRESS_MULTIPLIERS = _csv_floats(
    "BACKTEST_FEE_STRESS_MULTIPLIERS", "1.0,1.2"
)
BACKTEST_FUNDING_STRESS_MULTIPLIERS = _csv_floats(
    "BACKTEST_FUNDING_STRESS_MULTIPLIERS", "1.0,1.5,2.0"
)
BACKTEST_TP_CROSS_TICKS_GRID = _csv_ints(
    "BACKTEST_TP_CROSS_TICKS_GRID", "1,2"
)

BACKTEST_SUMMARY_FILE = os.getenv(
    "BACKTEST_SUMMARY_FILE", "state/backtest_v9_summary.csv"
)
BACKTEST_ANNUAL_FILE = os.getenv(
    "BACKTEST_ANNUAL_FILE", "state/backtest_v9_annual.csv"
)
BACKTEST_WALK_FORWARD_FILE = os.getenv(
    "BACKTEST_WALK_FORWARD_FILE", "state/backtest_v9_walk_forward.csv"
)
BACKTEST_OPPORTUNITIES_FILE = os.getenv(
    "BACKTEST_OPPORTUNITIES_FILE", "state/backtest_v9_opportunities.csv"
)
BACKTEST_TRADES_FILE = os.getenv(
    "BACKTEST_TRADES_FILE", "state/backtest_v9_trades.csv"
)
BACKTEST_PORTFOLIO_FILE = os.getenv(
    "BACKTEST_PORTFOLIO_FILE", "state/backtest_v9_portfolio.csv"
)
BACKTEST_RECOMMENDATION_FILE = os.getenv(
    "BACKTEST_RECOMMENDATION_FILE", "state/backtest_v9_recommendation.json"
)
BACKTEST_SENSITIVITY_FILE = os.getenv(
    "BACKTEST_SENSITIVITY_FILE", "state/backtest_v9_sensitivity.csv"
)
BACKTEST_STRESS_FILE = os.getenv(
    "BACKTEST_STRESS_FILE", "state/backtest_v9_stress.csv"
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
        raise ValueError("V9 requiere TAKE_PROFIT_ORDER_MODE=maker_limit.")
    if ENTRY_EXECUTION_MODE != "pullback_limit":
        raise ValueError("V9 mantiene ENTRY_EXECUTION_MODE=pullback_limit.")
    if ENTRY_MODE != "confirmed":
        raise ValueError("V9 mantiene ENTRY_MODE=confirmed.")
    if REGIME_MODE != "1h_sma200_4h_ema200":
        raise ValueError("V9 mantiene REGIME_MODE=1h_sma200_4h_ema200.")
    if ENTRY_LIMIT_OFFSET_BPS <= 0 or ENTRY_LIMIT_WAIT_SECONDS <= 0:
        raise ValueError("Offset/wait de entrada deben ser positivos.")
    if ENTRY_LIMIT_WAIT_SECONDS % 60 != 0:
        raise ValueError("ENTRY_LIMIT_WAIT_SECONDS debe ser multiplo de 60 para research 1m.")
    if ENTRY_FALLBACK_CAP_BPS < 0 or ENTRY_FALLBACK_MAX_LATENESS_SECONDS < 0:
        raise ValueError("Cap/lateness no pueden ser negativos.")
    if ENTRY_PENDING_POLL_SECONDS <= 0:
        raise ValueError("ENTRY_PENDING_POLL_SECONDS debe ser positivo.")
    if CANDLE_CLOSE_DELAY_SECONDS < 0:
        raise ValueError("CANDLE_CLOSE_DELAY_SECONDS no puede ser negativo.")
    if SIGNAL_MAX_AGE_SECONDS <= 0 or ERROR_RETRY_SECONDS <= 0:
        raise ValueError("SIGNAL_MAX_AGE_SECONDS/ERROR_RETRY_SECONDS deben ser positivos.")
    if POSITION_POLL_SECONDS <= 0:
        raise ValueError("POSITION_POLL_SECONDS debe ser positivo.")
    if not 0 < RISK_PER_TRADE <= 1:
        raise ValueError("RISK_PER_TRADE debe estar entre 0 y 1.")
    if not 0 < MAX_POSITION_PCT <= 1:
        raise ValueError("MAX_POSITION_PCT debe estar entre 0 y 1.")
    if not 0 <= CASH_BUFFER_PCT < 1:
        raise ValueError("CASH_BUFFER_PCT debe estar entre 0 y 1.")
    if not 0 < MAX_PORTFOLIO_RISK_PCT <= 1:
        raise ValueError("MAX_PORTFOLIO_RISK_PCT debe estar entre 0 y 1.")
    if not LIVE_SYMBOLS:
        raise ValueError("LIVE_SYMBOLS no puede estar vacio.")
    if LIVE_MAX_CONCURRENT_POSITIONS < 1:
        raise ValueError("LIVE_MAX_CONCURRENT_POSITIONS debe ser >=1.")
    if ATR_MULTIPLIER <= 0 or RISK_REWARD_RATIO <= 0:
        raise ValueError("ATR_MULTIPLIER/RISK_REWARD_RATIO deben ser positivos.")
    if MEAN_EXIT_Z < 0 or MAX_HOLDING_HOURS <= 0:
        raise ValueError("MEAN_EXIT_Z/MAX_HOLDING_HOURS invalidos.")
    if BREAK_EVEN_TRIGGER_R < 0:
        raise ValueError("BREAK_EVEN_TRIGGER_R no puede ser negativo.")
    if PARTIAL_TP_TRIGGER_R < 0:
        raise ValueError("PARTIAL_TP_TRIGGER_R no puede ser negativo.")
    if not 0 < PARTIAL_TP_FRACTION < 1:
        raise ValueError("PARTIAL_TP_FRACTION debe estar en (0,1).")
    if PARTIAL_TP_TRIGGER_R > 0 and EXIT_MODE not in {"fixed_rr", "fixed_rr_time", "hybrid_rr_mean", "hybrid_rr_mean_time"}:
        raise ValueError("Partial TP requiere un EXIT_MODE con TP fijo.")
    if BACKTEST_TAKER_FEE < 0 or BACKTEST_MAKER_FEE < 0:
        raise ValueError("Fees de backtest no pueden ser negativas.")
    if not BACKTEST_SYMBOLS:
        raise ValueError("BACKTEST_SYMBOLS no puede estar vacio.")
    if not BACKTEST_SELECTION_SYMBOLS:
        raise ValueError("BACKTEST_SELECTION_SYMBOLS no puede estar vacio.")
    unknown_selection = set(BACKTEST_SELECTION_SYMBOLS) - set(BACKTEST_SYMBOLS)
    if unknown_selection:
        raise ValueError(
            "BACKTEST_SELECTION_SYMBOLS debe ser subconjunto de BACKTEST_SYMBOLS: "
            f"{sorted(unknown_selection)}"
        )
    if not BACKTEST_MARKET_SLIPPAGE_BPS or any(v < 0 for v in BACKTEST_MARKET_SLIPPAGE_BPS):
        raise ValueError("BACKTEST_MARKET_SLIPPAGE_BPS debe contener valores >= 0.")
    if BACKTEST_TP_CROSS_TICKS < 1:
        raise ValueError("BACKTEST_TP_CROSS_TICKS debe ser >= 1.")
    if not BACKTEST_EXIT_PROFILES:
        raise ValueError("BACKTEST_EXIT_PROFILES no puede estar vacio.")
    if not BACKTEST_TP_RATIOS or any(v <= 0 for v in BACKTEST_TP_RATIOS):
        raise ValueError("BACKTEST_TP_RATIOS debe contener ratios > 0.")
    if not BACKTEST_MAX_POSITION_PCT_GRID or any(
        not 0 < v <= 1 for v in BACKTEST_MAX_POSITION_PCT_GRID
    ):
        raise ValueError("BACKTEST_MAX_POSITION_PCT_GRID debe estar en (0,1].")
    if not BACKTEST_RISK_PER_TRADE_GRID or any(
        not 0 < v <= 1 for v in BACKTEST_RISK_PER_TRADE_GRID
    ):
        raise ValueError("BACKTEST_RISK_PER_TRADE_GRID debe estar en (0,1].")
    if any(v not in {"long_only", "long_short"} for v in BACKTEST_SIDE_MODES):
        raise ValueError("BACKTEST_SIDE_MODES solo admite long_only,long_short.")
    if not 0 < BACKTEST_MAX_PORTFOLIO_RISK_PCT <= 1:
        raise ValueError("BACKTEST_MAX_PORTFOLIO_RISK_PCT debe estar en (0,1].")
    if BACKTEST_MAX_CONCURRENT_POSITIONS < 1:
        raise ValueError("BACKTEST_MAX_CONCURRENT_POSITIONS debe ser >=1.")
    if not 0 < BACKTEST_DD_LIMIT < 1:
        raise ValueError("BACKTEST_DD_LIMIT debe estar en (0,1).")
    if BACKTEST_MIN_POSITIVE_YEARS < 1:
        raise ValueError("BACKTEST_MIN_POSITIVE_YEARS debe ser positivo.")
    if BACKTEST_MIN_WORST_YEAR_PF < 0:
        raise ValueError("BACKTEST_MIN_WORST_YEAR_PF no puede ser negativo.")
    if not -1 < BACKTEST_MIN_WORST_YEAR_RETURN < 1:
        raise ValueError("BACKTEST_MIN_WORST_YEAR_RETURN debe estar en (-1,1).")
    if BACKTEST_MIN_GLOBAL_PF < 0:
        raise ValueError("BACKTEST_MIN_GLOBAL_PF no puede ser negativo.")
    if BACKTEST_MIN_TRAIN_TRADES < 1:
        raise ValueError("BACKTEST_MIN_TRAIN_TRADES debe ser positivo.")
    if not 0 < BACKTEST_PARTIAL_FRACTION < 1:
        raise ValueError("BACKTEST_PARTIAL_FRACTION debe estar en (0,1).")
    if BACKTEST_WORKERS < 0:
        raise ValueError("BACKTEST_WORKERS debe ser >= 0 (0 = auto).")
    if BACKTEST_SENSITIVITY_TOP_N < 1:
        raise ValueError("BACKTEST_SENSITIVITY_TOP_N debe ser >= 1.")
    if not BACKTEST_FEE_STRESS_MULTIPLIERS or any(v <= 0 for v in BACKTEST_FEE_STRESS_MULTIPLIERS):
        raise ValueError("BACKTEST_FEE_STRESS_MULTIPLIERS debe contener valores > 0.")
    if not BACKTEST_FUNDING_STRESS_MULTIPLIERS or any(v <= 0 for v in BACKTEST_FUNDING_STRESS_MULTIPLIERS):
        raise ValueError("BACKTEST_FUNDING_STRESS_MULTIPLIERS debe contener valores > 0.")
    if not BACKTEST_TP_CROSS_TICKS_GRID or any(v < 1 for v in BACKTEST_TP_CROSS_TICKS_GRID):
        raise ValueError("BACKTEST_TP_CROSS_TICKS_GRID debe contener enteros >= 1.")
