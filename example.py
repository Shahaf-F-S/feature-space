# example.py

import pandas as pd
import numpy as np

from feature_space import (
    Column, SMA, RSI, SuperTrend, ATR, Change, Momentum,
    MACDSignal, MACD, EMA, MACDHistogram, Dataset
)

df = pd.DataFrame(
    {
        key: np.random.random(100)
        for key in ('Open', 'High', 'Low', 'Close', 'Volume')
    }
)

print(df)

high = Column('High')
low = Column('Low')
close = Column('Close')

close_change = Change(close)
close_rsi_14 = RSI(close_change, 14)
close_momentum = Momentum(close_change, 35)

atr = ATR(high, low, close)
super_trend = SuperTrend(close, atr, 14, 3)

close_sma_20 = SMA(close, 20)
close_ema_34 = EMA(close, 34)
macd = MACD(close_ema_34, close_sma_20)
macd_signal_15 = MACDSignal(macd, 15)
macd_histogram = MACDHistogram(macd_signal_15)

print(df)

change_indicators = Dataset(
    name='Change_Features',
    features=[close_change, close_rsi_14, close_momentum]
)

change_indicators.calculate(df)

df.dropna(inplace=True)

print(df)

atr_indicators = Dataset(
    name='ATR_Features',
    features=[atr, super_trend]
)

atr_indicators.calculate(df)

df.dropna(inplace=True)

print(df)

macd_indicators = Dataset(
    name='MACD_Features',
    features=[
        close_sma_20, close_momentum,
        macd, macd_signal_15, macd_histogram
    ]
)

macd_indicators.calculate(df)

df.dropna(inplace=True)

print(df)