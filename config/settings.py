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

# Filtro de regimen V5. El live usa REGIME_MODE; el backtest compara baseline e intersecciones local+HTF.
REGIME_MODE = os.getenv("REGIME_MODE", "1h_sma200_4h_ema200").lower()
REGIME_TIMEFRAME = os.getenv("REGIME_TIMEFRAME", "4h")
REGIME_EMA_WINDOW = int(os.getenv("REGIME_EMA_WINDOW", "200"))
REGIME_SLOPE_LOOKBACK = int(os.getenv("REGIME_SLOPE_LOOKBACK", "6"))

# Tras los experimentos A/B/C, la baseline pasa a ser confirmacion + RR fijo.
ENTRY_MODE = os.getenv("ENTRY_MODE", "confirmed").lower()
EXIT_MODE = os.getenv("EXIT_MODE", "fixed_rr").lower()

# Solo se usa si EXIT_MODE=z_reversion.
MEAN_EXIT_Z = float(os.getenv("MEAN_EXIT_Z", "0.0"))
MAX_HOLDING_BARS = int(os.getenv("MAX_HOLDING_BARS", "48"))

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
# Costes / ejecucion de backtest V6
# -----------------------------------------------------------------------------
# Se mantienen por compatibilidad con scripts anteriores, pero V6 usa las
# variables BACKTEST_* de abajo para separar maker/taker y slippage por tipo
# de ejecucion.
TRADING_FEE = float(os.getenv("TRADING_FEE", "0.0005"))
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0005"))

# Supuestos de investigacion. Ajustalos a las comisiones efectivas de tu cuenta.
# Valores de referencia para usuario regular USD-M: maker 0.02%, taker 0.05%.
BACKTEST_TAKER_FEE = float(os.getenv("BACKTEST_TAKER_FEE", "0.0005"))
BACKTEST_MAKER_FEE = float(os.getenv("BACKTEST_MAKER_FEE", "0.0002"))

# Perfiles de ejecucion:
# - all_taker: entrada, stop y TP pagan taker; todos sufren slippage de mercado.
# - maker_tp: entrada y stop son taker; TP es LIMIT maker y no aplica slippage
#   de mercado en el modelo. El OHLC no modela cola/partial fills, por lo que
#   maker_tp es un escenario de ejecucion que debe validarse en Testnet/live.
BACKTEST_EXECUTION_PROFILES = tuple(
    value.strip().lower()
    for value in os.getenv(
        "BACKTEST_EXECUTION_PROFILES",
        "all_taker,maker_tp",
    ).split(",")
    if value.strip()
)

# Slippage adverso aplicado a ejecuciones de mercado, expresado en basis points.
BACKTEST_MARKET_SLIPPAGE_BPS = tuple(
    float(value.strip())
    for value in os.getenv(
        "BACKTEST_MARKET_SLIPPAGE_BPS",
        "0,1,2,3,5",
    ).split(",")
    if value.strip()
)

# Estrategia congelada para el estudio de costes V6.
BACKTEST_V6_REGIME_MODE = os.getenv(
    "BACKTEST_V6_REGIME_MODE",
    "1h_sma200_4h_ema200",
).lower()

# -----------------------------------------------------------------------------
# Ejecucion / persistencia
# -----------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
SIGNAL_MAX_AGE_SECONDS = int(os.getenv("SIGNAL_MAX_AGE_SECONDS", "300"))
STATE_FILE = os.getenv("STATE_FILE", "state/bot_state.json")

# V7 live execution: TP fijo como LIMIT reduceOnly post-only (maker).
# trigger_market conserva el comportamiento anterior por compatibilidad/migracion.
TAKE_PROFIT_ORDER_MODE = os.getenv(
    "TAKE_PROFIT_ORDER_MODE",
    "maker_limit",
).lower()
EXECUTION_JOURNAL_FILE = os.getenv(
    "EXECUTION_JOURNAL_FILE",
    "state/execution_journal.jsonl",
)

# -----------------------------------------------------------------------------
# Backtest V6: estrategia V5 congelada; se varia RR y modelo de ejecucion
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

# Modos que se comparan en el research grid.
# Valores validos: both,long,short
BACKTEST_SIDE_MODES = tuple(
    value.strip().lower()
    for value in os.getenv(
        "BACKTEST_SIDE_MODES",
        "long",
    ).split(",")
    if value.strip()
)

# Se conserva por compatibilidad con V5, pero V6 usa BACKTEST_V6_REGIME_MODE.
BACKTEST_REGIME_MODES = tuple(
    value.strip().lower()
    for value in os.getenv(
        "BACKTEST_REGIME_MODES",
        "1h_sma200_4h_ema200",
    ).split(",")
    if value.strip()
)

BACKTEST_YEARLY_BREAKDOWN = (
    os.getenv("BACKTEST_YEARLY_BREAKDOWN", "True").lower() == "true"
)
BACKTEST_WALK_FORWARD = (
    os.getenv("BACKTEST_WALK_FORWARD", "True").lower() == "true"
)
# Primer test anual despues de disponer al menos de N anos previos de historia.
BACKTEST_WALK_FORWARD_MIN_TRAIN_YEARS = int(
    os.getenv("BACKTEST_WALK_FORWARD_MIN_TRAIN_YEARS", "2")
)

# Compatibilidad con scripts anteriores; V3 ya no usa el corte 70/30 ni A/B/C.
BACKTEST_TRAIN_FRACTION = float(os.getenv("BACKTEST_TRAIN_FRACTION", "0.70"))
BACKTEST_COMPARE_VARIANTS = False


def validate_settings() -> None:
    if MARKET_MODE != "future":
        raise ValueError("El motor esta disenado para Binance USD-M Futures.")
    if POSITION_MODE != "one_way":
        raise ValueError("Esta version solo soporta POSITION_MODE='one_way'.")
    if MARGIN_MODE not in {"isolated", "cross"}:
        raise ValueError("MARGIN_MODE debe ser 'isolated' o 'cross'.")
    if not 1 <= LEVERAGE <= 125:
        raise ValueError("LEVERAGE debe estar entre 1 y 125.")
    if TRIGGER_PRICE_TYPE not in {"MARK_PRICE", "CONTRACT_PRICE"}:
        raise ValueError("TRIGGER_PRICE_TYPE debe ser MARK_PRICE o CONTRACT_PRICE.")
    if TAKE_PROFIT_ORDER_MODE not in {"maker_limit", "trigger_market"}:
        raise ValueError(
            "TAKE_PROFIT_ORDER_MODE debe ser maker_limit o trigger_market."
        )
    if not EXECUTION_JOURNAL_FILE.strip():
        raise ValueError("EXECUTION_JOURNAL_FILE no puede estar vacio.")
    if not ENABLE_LONGS and not ENABLE_SHORTS:
        raise ValueError("Debe estar habilitado al menos LONG o SHORT.")
    valid_regime_modes = {
        "1h_sma200",
        "4h_ema200",
        "4h_ema200_slope",
        "1h_sma200_4h_ema200",
        "1h_sma200_4h_ema200_slope",
    }
    if REGIME_MODE not in valid_regime_modes:
        raise ValueError(
            "REGIME_MODE no es un modo de regimen valido."
        )
    if REGIME_EMA_WINDOW <= 0:
        raise ValueError("REGIME_EMA_WINDOW debe ser positivo.")
    if REGIME_SLOPE_LOOKBACK <= 0:
        raise ValueError("REGIME_SLOPE_LOOKBACK debe ser positivo.")
    if ENTRY_MODE not in {"extreme", "confirmed"}:
        raise ValueError("ENTRY_MODE debe ser 'extreme' o 'confirmed'.")
    if EXIT_MODE not in {"fixed_rr", "z_reversion"}:
        raise ValueError("EXIT_MODE debe ser 'fixed_rr' o 'z_reversion'.")
    if MEAN_EXIT_Z < 0:
        raise ValueError("MEAN_EXIT_Z no puede ser negativo.")
    if MAX_HOLDING_BARS <= 0:
        raise ValueError("MAX_HOLDING_BARS debe ser positivo.")
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
    if TRADING_FEE < 0 or SLIPPAGE < 0:
        raise ValueError("TRADING_FEE y SLIPPAGE no pueden ser negativos.")
    if BACKTEST_TAKER_FEE < 0 or BACKTEST_MAKER_FEE < 0:
        raise ValueError("BACKTEST_TAKER_FEE y BACKTEST_MAKER_FEE no pueden ser negativos.")
    valid_execution_profiles = {"all_taker", "maker_tp"}
    if not BACKTEST_EXECUTION_PROFILES or any(
        value not in valid_execution_profiles for value in BACKTEST_EXECUTION_PROFILES
    ):
        raise ValueError(
            "BACKTEST_EXECUTION_PROFILES solo puede contener: all_taker,maker_tp."
        )
    if not BACKTEST_MARKET_SLIPPAGE_BPS or any(
        value < 0 for value in BACKTEST_MARKET_SLIPPAGE_BPS
    ):
        raise ValueError(
            "BACKTEST_MARKET_SLIPPAGE_BPS debe contener valores >= 0."
        )
    if not BACKTEST_TP_RATIOS or any(value <= 0 for value in BACKTEST_TP_RATIOS):
        raise ValueError("BACKTEST_TP_RATIOS debe contener ratios positivos.")
    valid_side_modes = {"both", "long", "short"}
    if not BACKTEST_SIDE_MODES or any(
        value not in valid_side_modes for value in BACKTEST_SIDE_MODES
    ):
        raise ValueError(
            "BACKTEST_SIDE_MODES solo puede contener: both,long,short."
        )
    if not BACKTEST_REGIME_MODES or any(
        value not in valid_regime_modes for value in BACKTEST_REGIME_MODES
    ):
        raise ValueError(
            "BACKTEST_REGIME_MODES solo puede contener: "
            "1h_sma200,1h_sma200_4h_ema200,1h_sma200_4h_ema200_slope "
            "(los modos 4h puros se conservan por compatibilidad)."
        )
    if BACKTEST_V6_REGIME_MODE not in valid_regime_modes:
        raise ValueError("BACKTEST_V6_REGIME_MODE no es un modo de regimen valido.")
    if BACKTEST_WALK_FORWARD_MIN_TRAIN_YEARS < 1:
        raise ValueError("BACKTEST_WALK_FORWARD_MIN_TRAIN_YEARS debe ser >= 1.")
