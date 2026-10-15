import time
import pandas as pd
from src.data.fetcher import fetch_binance_klines
from src.strategy.mean_reversion import MeanReversionStrategy
from src.execution.engine import BinanceExecutionEngine  # Ajusta la ruta si es necesario
from config.settings import (
    SYMBOL, TIMEFRAME, LOOKBACK_WINDOW, Z_THRESHOLD, 
    TREND_WINDOW, ATR_WINDOW, ATR_MULTIPLIER, RISK_REWARD_RATIO, INITIAL_CAPITAL
)

def main():
    print(f"🚀 Iniciando Bot de Reversión a la Media en {SYMBOL} ({TIMEFRAME})...")
    
    # Inicializar estrategia y motor de ejecución
    strategy = MeanReversionStrategy(
        window=LOOKBACK_WINDOW, 
        z_threshold=Z_THRESHOLD,
        trend_window=TREND_WINDOW,
        atr_window=ATR_WINDOW,
        atr_multiplier=ATR_MULTIPLIER,
        risk_reward_ratio=RISK_REWARD_RATIO
    )
    engine = BinanceExecutionEngine()

    try:
        while True:
            # 1. Obtener datos recientes
            df = fetch_binance_klines(symbol=SYMBOL, timeframe=TIMEFRAME, limit=300)
            df_signals = strategy.calculate_signals(df)
            latest = df_signals.iloc[-1]
            
            print(f"[{latest.name}] Close: {latest['close']:.2f} | Z-Score: {latest['z_score']:.2f} | Trend SMA: {latest['sma_trend']:.2f}")

            # 2. Comprobar si ya tenemos una posición abierta para no solapar
            position = engine.get_current_position(SYMBOL)
            
            if not position['has_position']:
                # 3. Evaluar señales de entrada (asumiendo que entry_long / entry_short son booleanos)
                # Si tu estrategia devuelve un campo 'signal' numérico (1, -1, 0), adaptamos la condición:
                is_long = latest.get('entry_long', False) or latest.get('signal', 0) == 1
                is_short = latest.get('entry_short', False) or latest.get('signal', 0) == -1

                if is_long or is_short:
                    side = 'buy' if is_long else 'sell'
                    entry_price = latest['close']
                    atr_value = latest['atr']
                    
                    # Calcular niveles de SL y TP
                    if side == 'buy':
                        sl_price = entry_price - (atr_value * ATR_MULTIPLIER)
                        tp_price = entry_price + ((entry_price - sl_price) * RISK_REWARD_RATIO)
                    else:
                        sl_price = entry_price + (atr_value * ATR_MULTIPLIER)
                        tp_price = entry_price - ((sl_price - entry_price) * RISK_REWARD_RATIO)

                    # Calcular tamaño de posición (ej. arriesgando un porcentaje fijo o capital base de prueba)
                    # Para pruebas en Testnet, podemos usar un monto fijo en USDT o derivado del balance libre
                    usdt_balance = engine.get_balance('USDT')
                    allocation_usdt = min(usdt_balance * 0.1, 100) # Usar 10% del balance o máximo 100 USDT para test
                    amount = round(allocation_usdt / entry_price, 5)

                    print(f"🎯 Señal detectada [{side.upper()}]. Precio: {entry_price} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
                    
                    # Ejecutar en Testnet
                    engine.execute_trade_with_sl_tp(
                        symbol=SYMBOL,
                        side=side,
                        amount=amount,
                        entry_price=entry_price,
                        sl_price=sl_price,
                        tp_price=tp_price
                    )
            else:
                print("🛡️ Posición ya activa en el mercado. Monitorizando...")

            # Esperar antes de la siguiente comprobación (en velas de 1h lo ideal es chequear cada pocos minutos o al cierre de vela)
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido por el usuario.")

if __name__ == "__main__":
    main()