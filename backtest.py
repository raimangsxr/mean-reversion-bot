import ccxt
import pandas as pd
import vectorbt as vbt
from config.settings import *
from src.strategy.mean_reversion import MeanReversionStrategy

def run_backtest():
    print(f"Descargando datos históricos para {SYMBOL} ({TIMEFRAME}) vía CCXT...")
    
    # Inicializar exchange con ccxt (público, no requiere claves para histórico)
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Obtener timestamps para las fechas deseadas
    since = exchange.parse8601('2026-04-18T00:00:00Z')
    end_ts = exchange.parse8601('2026-10-15T00:00:00Z')
    
    all_ohlcv = []
    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=1000)
        if not ohlcv:
            break
        since = ohlcv[-1][0] + 1
        all_ohlcv.extend(ohlcv)
        if len(ohlcv) < 1000:
            break
            
    # Convertir a DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Filtrar estrictamente hasta la fecha de fin
    df = df[df.index <= '2026-10-15 19:00:00']

    # Instanciar y calcular estrategia
    strategy = MeanReversionStrategy(
        window=LOOKBACK_WINDOW,
        z_threshold=Z_THRESHOLD,
        trend_window=TREND_WINDOW,
        atr_window=ATR_WINDOW,
        atr_multiplier=ATR_MULTIPLIER,
        risk_reward_ratio=RISK_REWARD_RATIO
    )
    
    df_signals = strategy.calculate_signals(df)
    
    # Calcular distancias porcentuales para SL y TP basadas en ATR
    atr_pct = (df_signals['atr'] * ATR_MULTIPLIER) / df_signals['close']
    tp_pct = atr_pct * RISK_REWARD_RATIO

    # Simular Portfolio con vectorbt incluyendo TP y SL dinámicos
    portfolio = vbt.Portfolio.from_signals(
        close=df_signals['close'],
        high=df_signals['high'],
        low=df_signals['low'],
        entries=df_signals['entry_long'],
        short_entries=df_signals['entry_short'],
        sl_stop=atr_pct,          
        tp_stop=tp_pct,           
        init_cash=INITIAL_CAPITAL,
        fees=TRADING_FEE,
        slippage=SLIPPAGE
    )
    
    # Imprimir reporte de rendimiento
    print("\n📊 --- RESULTADOS DEL BACKTEST CON SL/TP ATR ---")
    print(portfolio.stats())

if __name__ == "__main__":
    run_backtest()