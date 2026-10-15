import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"

# Money settings
INITIAL_CAPITAL = 10000.0
TRADING_FEE = 0.001  # 0.1% de comisión
SLIPPAGE = 0.0005    # 0.05% de slippage estimado


SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LOOKBACK_WINDOW = 20
Z_THRESHOLD = 2.2      # <--- Ajustado a 2.2 para mayor flexibilidad
TREND_WINDOW = 200     # <--- Filtro de régimen macro
ATR_WINDOW = 14
ATR_MULTIPLIER = 1.5   # <--- Distancia del Stop Loss
RISK_REWARD_RATIO = 1.8 # <--- Take Profit a 1.5x del Stop Loss