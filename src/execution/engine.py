import ccxt
import pandas as pd
from config.settings import BINANCE_API_KEY, BINANCE_SECRET_KEY, USE_TESTNET

class BinanceExecutionEngine:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'enableRateLimit': True,
        })
        if USE_TESTNET:
            self.exchange.set_sandbox_mode(True)
            
        # Cargar mercados para asegurar precisión en decimales y tamaños mínimos
        self.exchange.load_markets()

    def get_balance(self, currency: str = 'USDT') -> float:
        """Devuelve el balance libre disponible para operar de una moneda."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['free'].get(currency, 0.0))
        except Exception as e:
            print(f"❌ Error al obtener el balance: {e}")
            return 0.0

    def get_current_position(self, symbol: str) -> dict:
        """
        Consulta si hay saldo o posición abierta en el par (para Spot o Futures según configuración).
        En Spot, revisa si tenemos saldo del activo base superior al polvo de mercado.
        """
        base_currency = symbol.split('/')[0]
        try:
            balance = self.exchange.fetch_balance()
            free_base = float(balance['free'].get(base_currency, 0.0))
            total_base = float(balance['total'].get(base_currency, 0.0))
            return {
                'has_position': total_base > 0.0001, # Ajustar umbral según activo
                'free': free_base,
                'total': total_base
            }
        except Exception as e:
            print(f"❌ Error al comprobar posición para {symbol}: {e}")
            return {'has_position': False, 'free': 0.0, 'total': 0.0}

    def create_order(self, symbol: str, type: str, side: str, amount: float, price: float = None):
        """Envía una orden estándar al exchange."""
        try:
            order = self.exchange.create_order(symbol, type, side, amount, price)
            print(f"✅ Orden ejecutada con éxito: {order['id']} - {side.upper()} {amount} {symbol}")
            return order
        except Exception as e:
            print(f"❌ Error al crear la orden {side} en {symbol}: {e}")
            return None

    def execute_trade_with_sl_tp(self, symbol: str, side: str, amount: float, entry_price: float, sl_price: float, tp_price: float):
        """
        Ejecuta una entrada a mercado y programa sus niveles de Stop Loss y Take Profit
        mediante órdenes condicionales o OCO soportadas por la API de Binance.
        """
        try:
            # 1. Ejecutar orden de entrada a mercado
            print(f"🚀 Lanzando entrada {side.upper()} para {symbol} (Cantidad: {amount})...")
            market_order = self.exchange.create_order(symbol, 'market', side, amount)
            
            # Determinar el lado opuesto para las órdenes de salida
            exit_side = 'sell' if side == 'buy' else 'buy'
            
            # 2. Configurar Stop Loss (Stop Loss Limit o Stop Market según preferencia)
            # Nota: En Binance Spot, las órdenes stop loss se gestionan habitualmente con 'STOP_LOSS_LIMIT' o 'STOP_MARKET'
            sl_params = {
                'stopPrice': self.exchange.price_to_precision(symbol, sl_price),
            }
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=exit_side,
                amount=amount,
                params=sl_params
            )
            print(f"🛡️ Stop Loss programado en: {sl_price}")

            # 3. Configurar Take Profit (Limit Order)
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type='LIMIT',
                side=exit_side,
                amount=amount,
                price=tp_price
            )
            print(f"🎯 Take Profit programado en: {tp_price}")

            return {
                'entry': market_order,
                'sl': sl_order,
                'tp': tp_order
            }
            
        except Exception as e:
            print(f"❌ Error crítico ejecutando la operativa con SL/TP en {symbol}: {e}")
            return None