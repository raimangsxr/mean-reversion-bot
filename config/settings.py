import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"

# -----------------------------------------------------------------------------
# Mercado: Binance USD-M perpetual futures
# -----------------------------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "BTC/USDT:USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
MARKET_MODE = "future"
ENABLE_LONGS = os.getenv("ENABLE_LONGS", "True").lower() == "true"
ENABLE_SHORTS = os.getenv("ENABLE_SHORTS", "False").lower() == "true"

# One-way Mode: una sola posicion neta por simbolo.
POSITION_MODE = "one_way"
MARGIN_MODE = os.getenv("MARGIN_MODE", "isolated").lower()
LEVERAGE = int(os.getenv("LEVERAGE", "2"))
TRIGGER_PRICE_TYPE = os.getenv("TRIGGER_PRICE_TYPE", "MARK_PRICE").upper()

# -----------------------------------------------------------------------------
# Estrategia live
# -----------------------------------------------------------------------------
LOOKBACK_WINDOW = int(os.getenv("LOOKBACK_WINDOW", "20"))
Z_THRESHOLD = float(os.getenv("Z_THRESHOLD", "2.2"))
TREND_WINDOW = int(os.getenv("TREND_WINDOW", "200"))
ATR_WINDOW = int(os.getenv("ATR_WINDOW", "14"))

# Filtro de regimen de la estrategia V8 final.
REGIME_MODE = os.getenv("REGIME_MODE", "1h_sma200_4h_ema200").lower()
REGIME_TIMEFRAME = os.getenv("REGIME_TIMEFRAME", "4h")
REGIME_EMA_WINDOW = int(os.getenv("REGIME_EMA_WINDOW", "200"))
REGIME_SLOPE_LOOKBACK = int(os.getenv("REGIME_SLOPE_LOOKBACK", "6"))

# Estrategia V8 final: entrada confirmada y salida por RR fijo.
ENTRY_MODE = os.getenv("ENTRY_MODE", "confirmed").lower()
EXIT_MODE = os.getenv("EXIT_MODE", "fixed_rr").lower()

ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))
# Valor live. El barrido de backtest NO modifica este parametro.
RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "1.5"))

# -----------------------------------------------------------------------------
# Riesgo / sizing
# -----------------------------------------------------------------------------
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.005"))  # 0.5%
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10"))  # 10%
CASH_BUFFER_PCT = float(os.getenv("CASH_BUFFER_PCT", "0.05"))  # 5%

# -----------------------------------------------------------------------------
# Costes / ejecucion del backtest V8 final
# -----------------------------------------------------------------------------
# Supuestos de investigacion. Ajustalos a las comisiones efectivas de tu cuenta.
BACKTEST_TAKER_FEE = float(os.getenv("BACKTEST_TAKER_FEE", "0.0005"))
BACKTEST_MAKER_FEE = float(os.getenv("BACKTEST_MAKER_FEE", "0.0002"))

# Slippage adverso aplicado a ejecuciones de mercado, expresado en basis points.
BACKTEST_MARKET_SLIPPAGE_BPS = tuple(
    float(value.strip())
    for value in os.getenv(
        "BACKTEST_MARKET_SLIPPAGE_BPS",
        "0,1,2,3,5",
    ).split(",")
    if value.strip()
)

# -----------------------------------------------------------------------------
# Ejecucion / persistencia
# -----------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
ERROR_RETRY_SECONDS = int(os.getenv("ERROR_RETRY_SECONDS", "10"))
SIGNAL_MAX_AGE_SECONDS = int(os.getenv("SIGNAL_MAX_AGE_SECONDS", "60"))

# V8 final: LIMIT -3bp anclada al close de la vela de senal,
# espera 7m y fallback MARKET solo si el precio sigue dentro de +5bp respecto
# al precio observado al iniciar (t=0).
ENTRY_EXECUTION_MODE = os.getenv(
    "ENTRY_EXECUTION_MODE",
    "pullback_limit",
).lower()
ENTRY_LIMIT_OFFSET_BPS = float(os.getenv("ENTRY_LIMIT_OFFSET_BPS", "3.0"))
ENTRY_LIMIT_WAIT_SECONDS = int(os.getenv("ENTRY_LIMIT_WAIT_SECONDS", "420"))
ENTRY_FALLBACK_CAP_BPS = float(os.getenv("ENTRY_FALLBACK_CAP_BPS", "5.0"))
ENTRY_PENDING_POLL_SECONDS = int(os.getenv("ENTRY_PENDING_POLL_SECONDS", "10"))
# Si el proceso despierta demasiado tarde, no ejecutamos un fallback que ya no
# representa el experimento de 7 minutos. Se cancela la senal.
ENTRY_FALLBACK_MAX_LATENESS_SECONDS = int(
    os.getenv("ENTRY_FALLBACK_MAX_LATENESS_SECONDS", "60")
)

STATE_FILE = os.getenv("STATE_FILE", "state/bot_state.json")

# V8 final: TP fijo como LIMIT reduceOnly post-only (maker).
TAKE_PROFIT_ORDER_MODE = os.getenv(
    "TAKE_PROFIT_ORDER_MODE",
    "maker_limit",
).lower()
EXECUTION_JOURNAL_FILE = os.getenv(
    "EXECUTION_JOURNAL_FILE",
    "state/execution_journal.jsonl",
)

# -----------------------------------------------------------------------------
# Backtest V8 final: control MARKET inmediato vs politica pullback/no-chase; grid RR/slippage
# -----------------------------------------------------------------------------
BACKTEST_START = os.getenv("BACKTEST_START", "2020-01-01T00:00:00Z")
BACKTEST_END = os.getenv(
    "BACKTEST_END",
    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
)

# Rejilla deliberadamente gruesa: buscamos una meseta robusta, no el mejor
# decimal concreto.
BACKTEST_TP_RATIOS = tuple(
    float(value.strip())
    for value in os.getenv(
        "BACKTEST_TP_RATIOS",
        "1.5,1.8",
    ).split(",")
    if value.strip()
)

# Modos de entrada que compara el backtest principal.
# - market_0m: control MARKET inmediato + TP maker.
# - pullback_limit: entrada V8 final, LIMIT -offset / wait / cap no-chase.
BACKTEST_ENTRY_EXECUTION_MODES = tuple(
    value.strip().lower()
    for value in os.getenv(
        "BACKTEST_ENTRY_EXECUTION_MODES",
        "market_0m,pullback_limit",
    ).split(",")
    if value.strip()
)

# Cache 1m reutilizable. El backtest descarga solo las horas posteriores a
# senales candidatas que falten en este fichero.
BACKTEST_MINUTE_CACHE_FILE = os.getenv(
    "BACKTEST_MINUTE_CACHE_FILE",
    "state/backtest_1m_cache.csv",
)
BACKTEST_SUMMARY_FILE = os.getenv(
    "BACKTEST_SUMMARY_FILE",
    "state/backtest_v8_summary.csv",
)
BACKTEST_ANNUAL_FILE = os.getenv(
    "BACKTEST_ANNUAL_FILE",
    "state/backtest_v8_annual.csv",
)
BACKTEST_TRADES_FILE = os.getenv(
    "BACKTEST_TRADES_FILE",
    "state/backtest_v8_trades.csv",
)



def validate_settings() -> None:
    if MARKET_MODE != "future":
        raise ValueError("El motor esta disenado para Binance USD-M Futures.")
    if POSITION_MODE != "one_way":
        raise ValueError("Esta version solo soporta POSITION_MODE='one_way'.")
    if MARGIN_MODE != "isolated":
        raise ValueError("V8 final requiere MARGIN_MODE=isolated.")
    if not 1 <= LEVERAGE <= 125:
        raise ValueError("LEVERAGE debe estar entre 1 y 125.")
    if TRIGGER_PRICE_TYPE not in {"MARK_PRICE", "CONTRACT_PRICE"}:
        raise ValueError("TRIGGER_PRICE_TYPE debe ser MARK_PRICE o CONTRACT_PRICE.")
    if TAKE_PROFIT_ORDER_MODE != "maker_limit":
        raise ValueError("V8 final requiere TAKE_PROFIT_ORDER_MODE=maker_limit.")
    if not EXECUTION_JOURNAL_FILE.strip():
        raise ValueError("EXECUTION_JOURNAL_FILE no puede estar vacio.")
    if POLL_INTERVAL_SECONDS <= 0:
        raise ValueError("POLL_INTERVAL_SECONDS debe ser positivo.")
    if ERROR_RETRY_SECONDS <= 0:
        raise ValueError("ERROR_RETRY_SECONDS debe ser positivo.")
    if SIGNAL_MAX_AGE_SECONDS <= 0:
        raise ValueError("SIGNAL_MAX_AGE_SECONDS debe ser positivo.")
    if ENTRY_EXECUTION_MODE != "pullback_limit":
        raise ValueError("V8 final requiere ENTRY_EXECUTION_MODE=pullback_limit.")
    if ENTRY_LIMIT_OFFSET_BPS <= 0:
        raise ValueError("ENTRY_LIMIT_OFFSET_BPS debe ser positivo.")
    if ENTRY_LIMIT_WAIT_SECONDS <= 0:
        raise ValueError("ENTRY_LIMIT_WAIT_SECONDS debe ser positivo.")
    if ENTRY_FALLBACK_CAP_BPS < 0:
        raise ValueError("ENTRY_FALLBACK_CAP_BPS no puede ser negativo.")
    if ENTRY_PENDING_POLL_SECONDS <= 0:
        raise ValueError("ENTRY_PENDING_POLL_SECONDS debe ser positivo.")
    if ENTRY_PENDING_POLL_SECONDS > ENTRY_LIMIT_WAIT_SECONDS:
        raise ValueError(
            "ENTRY_PENDING_POLL_SECONDS no puede superar ENTRY_LIMIT_WAIT_SECONDS."
        )
    if ENTRY_FALLBACK_MAX_LATENESS_SECONDS < 0:
        raise ValueError(
            "ENTRY_FALLBACK_MAX_LATENESS_SECONDS no puede ser negativo."
        )
    if not ENABLE_LONGS or ENABLE_SHORTS:
        raise ValueError("V8 final requiere ENABLE_LONGS=True y ENABLE_SHORTS=False.")
    if REGIME_MODE != "1h_sma200_4h_ema200":
        raise ValueError("V8 final requiere REGIME_MODE=1h_sma200_4h_ema200.")
    if REGIME_EMA_WINDOW <= 0:
        raise ValueError("REGIME_EMA_WINDOW debe ser positivo.")
    if REGIME_SLOPE_LOOKBACK <= 0:
        raise ValueError("REGIME_SLOPE_LOOKBACK debe ser positivo.")
    if ENTRY_MODE != "confirmed":
        raise ValueError("V8 final requiere ENTRY_MODE=confirmed.")
    if EXIT_MODE != "fixed_rr":
        raise ValueError("V8 final requiere EXIT_MODE=fixed_rr.")
    if not 0 < RISK_PER_TRADE <= 1:
        raise ValueError("RISK_PER_TRADE debe estar entre 0 y 1.")
    if not 0 < MAX_POSITION_PCT <= 1:
        raise ValueError("MAX_POSITION_PCT debe estar entre 0 y 1.")
    if not 0 <= CASH_BUFFER_PCT < 1:
        raise ValueError("CASH_BUFFER_PCT debe estar entre 0 y 1.")
    if RISK_REWARD_RATIO <= 0:
        raise ValueError("RISK_REWARD_RATIO debe ser positivo.")
    if ATR_MULTIPLIER <= 0:
        raise ValueError("ATR_MULTIPLIER debe ser positivo.")
    if BACKTEST_TAKER_FEE < 0 or BACKTEST_MAKER_FEE < 0:
        raise ValueError("BACKTEST_TAKER_FEE y BACKTEST_MAKER_FEE no pueden ser negativos.")
    if not BACKTEST_MARKET_SLIPPAGE_BPS or any(
        value < 0 for value in BACKTEST_MARKET_SLIPPAGE_BPS
    ):
        raise ValueError(
            "BACKTEST_MARKET_SLIPPAGE_BPS debe contener valores >= 0."
        )
    if not BACKTEST_TP_RATIOS or any(value <= 0 for value in BACKTEST_TP_RATIOS):
        raise ValueError("BACKTEST_TP_RATIOS debe contener ratios positivos.")
    valid_entry_execution_modes = {"market_0m", "pullback_limit"}
    if not BACKTEST_ENTRY_EXECUTION_MODES or any(
        value not in valid_entry_execution_modes
        for value in BACKTEST_ENTRY_EXECUTION_MODES
    ):
        raise ValueError(
            "BACKTEST_ENTRY_EXECUTION_MODES solo puede contener: "
            "market_0m,pullback_limit."
        )
    if not BACKTEST_MINUTE_CACHE_FILE.strip():
        raise ValueError("BACKTEST_MINUTE_CACHE_FILE no puede estar vacio.")
