import pandas as pd
import numpy as np

class MeanReversionStrategy:
    def __init__(self, window: int = 20, z_threshold: float = 2.2, trend_window: int = 200, 
                 atr_window: int = 14, atr_multiplier: float = 2.0, risk_reward_ratio: float = 1.5):
        self.window = window
        self.z_threshold = z_threshold
        self.trend_window = trend_window
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_window).mean()

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Indicadores base
        df['sma'] = df['close'].rolling(window=self.window).mean()
        df['std'] = df['close'].rolling(window=self.window).std()
        df['z_score'] = (df['close'] - df['sma']) / df['std']
        df['sma_trend'] = df['close'].rolling(window=self.trend_window).mean()
        df['atr'] = self.calculate_atr(df)
        
        # Inicializar señales booleanas para vectorbt
        df['entry_long'] = False
        df['entry_short'] = False
        
        # Filtro de régimen + Z-Score moderado (2.2)
        long_condition = (df['z_score'] < -self.z_threshold) & (df['close'] > df['sma_trend'])
        short_condition = (df['z_score'] > self.z_threshold) & (df['close'] < df['sma_trend'])
        
        df.loc[long_condition, 'entry_long'] = True
        df.loc[short_condition, 'entry_short'] = True
        
        return df