# features.py

import numpy as np
import pandas as pd

from feature_space.feature import Feature, Column

__all__ = (
    'EMA',
    'Flips',
    'MACD',
    'MACDSignal',
    'MACDHistogram',
    'Change',
    'Momentum',
    'MomentumOscillator',
    'MiddleBollingerBand',
    'BottomBollingerBand',
    'SMA',
    'TRAMA',
    'STD',
    'SuperTrend',
    'LiquiditySpikes',
    'RSI',
    'TopBollingerBand',
    'ATR',
    'Volatility'
)

CLOSE, HIGH, LOW = 'Close', 'High', 'Low'

def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:

    atr_data = pd.DataFrame(
        {HIGH: high, LOW: low, CLOSE: close},
        index=close.index
    )

    atr_data['High_Low'] = atr_data[HIGH] - atr_data[LOW]
    atr_data['High_Previous_Close'] = abs(atr_data[HIGH] - atr_data[CLOSE].shift())
    atr_data['Low_Previous_Close'] = abs(atr_data[LOW] - atr_data[CLOSE].shift())

    return atr_data[
        ['High_Low', 'High_Previous_Close', 'Low_Previous_Close']
    ].max(axis=1).mean()

def super_trend(
        data: pd.Series,
        atr: pd.Series,
        span: int = 10,
        factor: int = 3
) -> pd.Series:

    rolling_mean = data.rolling(span).mean()
    rolling_mean = rolling_mean.fillna(rolling_mean.iloc[span - 1])

    upper_band = rolling_mean + (factor * atr)
    lower_band = rolling_mean - (factor * atr)

    conditions = [data > upper_band, data < lower_band]

    return pd.Series(
        np.select(conditions, [1, -1], default=0),
        index=data.index
    )

def liquidity_spikes(
        data: pd.Series,
        span: int = 20,
        z_score_threshold: float = 2.0,
        gradual: bool = False
) -> pd.Series:

    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    rolling_average = data.rolling(span, min_periods=1).mean()
    rolling_std = data.rolling(span, min_periods=1).std()

    rolling_average = rolling_average.fillna(0)
    rolling_std = rolling_std.fillna(0)

    z_scores = (data - rolling_average) / rolling_std

    if gradual:
        abnormal_spikes = z_scores
        abnormal_spikes.fillna(abnormal_spikes.iloc[1], inplace=True)

    else:
        abnormal_spikes = pd.Series(
            np.where(z_scores > z_score_threshold, 1, 0),
            index=data.index
        )

        abnormal_spikes.astype(int)

    return abnormal_spikes

class Change(Feature):

    def __init__(self, feature: Feature, name: str = None) -> None:

        self.feature = feature

        super().__init__(
            name=name or f'{self.feature.name}_Change',
            features=[self.feature],
            calculator=lambda f: self.feature.result.diff()
        )

class MiddleBollingerBand(Feature):

    def __init__(self, feature: Feature, span: int, name: str = None) -> None:

        self.feature = feature
        self.span = span

        self.sma = SMA(self.feature, span=self.span)

        super().__init__(
            name=name or f'{self.feature.name}_Middle_Bollinger_Band_{span}',
            features=[self.sma],
            calculator=lambda f: self.sma.result
        )

class STD(Feature):

    def __init__(self, feature: Feature, span: int, name: str = None) -> None:

        self.feature = feature
        self.span = span

        super().__init__(
            name=name or f'{feature.name}_STD_{self.span}',
            kwargs=dict(span=span),
            features=[self.feature],
            calculator=lambda f: (
                self.feature.result.rolling(window=span).std()
            )
        )

class BottomBollingerBand(Feature):

    def __init__(self, band: MiddleBollingerBand, std: STD, name: str = None) -> None:

        self.band = band
        self.std = std

        super().__init__(
            name=name or f'{self.std.feature.name}_Bottom_Bollinger_Band',
            features=[self.band, self.std],
            calculator=lambda f: self.band.result - (2 * self.std.result)
        )

class TopBollingerBand(Feature):

    def __init__(self, band: MiddleBollingerBand, std: STD, name: str = None) -> None:

        self.band = band
        self.std = std

        super().__init__(
            name=name or f'{self.std.feature.name}_Top_Bollinger_Band',
            features=[self.band, self.std],
            calculator=lambda f: self.band.result + (2 * self.std.result)
        )

class Volatility(Feature):

    def __init__(self, change: Change, name: str = None) -> None:

        self.change = change

        super().__init__(
            name=name or f'{self.change.feature.name}_Volatility',
            features=[self.change],
            calculator=lambda f: self.change.result.abs()
        )

class TRAMA(Feature):

    def __init__(self, volatility: Volatility, span: int, name: str = None) -> None:

        self.volatility = volatility
        self.span = span

        self.sma = SMA(self.volatility.change.feature, span=self.span)

        super().__init__(
            name=name or f'{self.volatility.name}_TRAMA_{self.span}',
            features=[self.volatility, self.sma],
            calculator=lambda f: (
                self.sma.result + (self.volatility.result * 0.1)
            )
        )

class Momentum(Feature):

    def __init__(self, change: Change, span: int, name: str = None) -> None:

        self.change = change
        self.span = span

        super().__init__(
            name=name or f'{self.change.feature.name}_Momentum_{self.span}',
            features=[self.change],
            calculator=lambda f: (
                self.change.result.rolling(window=self.span, min_periods=1).sum()
            )
        )

class MomentumOscillator(Feature):

    def __init__(self, feature: Feature, span: int, name: str = None) -> None:

        self.feature = feature
        self.span = span

        super().__init__(
            name=name or f'{self.feature.name}_Momentum_Oscillator_{self.span}',
            features=[self.feature],
            calculator=lambda f: (
                (
                    self.feature.result.diff(self.span) /
                    self.feature.result.shift(self.span)
                ) * 100
            )
        )

class RSI(Feature):

    def __init__(self, change: Change, span: int, name: str = None) -> None:

        self.change = change
        self.span = span

        super().__init__(
            name=name or f'{self.change.feature.name}_RSI_{self.span}',
            features=[self.change],
            calculator=lambda f: (
                (gain := self.change.result.apply(lambda x: x if x > 0 else 0)),
                (loss := self.change.result.apply(lambda x: abs(x) if x < 0 else 0)),
                (avg_gain := gain.rolling(window=self.span).mean()),
                (avg_loss := loss.rolling(window=self.span).mean()),
                (relative_strength := avg_gain / avg_loss),
                100 - (100 / (1 + relative_strength))
            )[-1]
        )

class EMA(Feature):

    def __init__(self, feature: Feature, span: int, name: str = None) -> None:

        self.feature = feature
        self.span = span

        super().__init__(
            name=name or f'{feature.name}_EMA_{self.span}',
            features=[self.feature],
            calculator=lambda f: (
                self.feature.result.ewm(span=self.span, adjust=False).mean()
            )
        )

class SMA(Feature):

    def __init__(self, feature: Feature, span: int, name: str = None) -> None:

        self.feature = feature
        self.span = span

        super().__init__(
            name=name or f'{self.feature.name}_SMA_{self.span}',
            features=[self.feature],
            calculator=lambda f: (
                self.feature.result.rolling(window=self.span).mean()
            )
        )

class MACD(Feature):

    def __init__(self, f1: Feature, f2: Feature, name: str = None) -> None:

        self.f1 = f1
        self.f2 = f2

        super().__init__(
            name=name or f'{self.f1.name}_{self.f2.name}_MACD',
            features=[self.f1, self.f2],
            calculator=lambda f: self.f1.result - self.f2.result
        )

class Flips(Feature):

    def __init__(self, f1: Feature, f2: Feature, name: str = None) -> None:

        self.f1 = f1
        self.f2 = f2

        super().__init__(
            name=name or f'{self.f1.name}_{self.f2.name}_Flips',
            features=[self.f1, self.f1],
            calculator=lambda f: (
                pd.Series(
                    (self.f1.result > self.f2.result) !=
                    (self.f1.result.shift(1) > self.f2.result.shift(1))
                ).astype(int)
            )
        )

class MACDSignal(Feature):

    def __init__(self, macd: MACD, span: int, name: str = None) -> None:

        self.macd = macd
        self.span = span

        super().__init__(
            name=name or f"{self.macd.name}_Signal_{self.span}",
            features=[self.macd],
            calculator=lambda f: (
                self.macd.result.rolling(window=self.span, min_periods=1).mean()
            )
        )

class MACDHistogram(Feature):

    def __init__(self, macd_signal: MACDSignal, name: str = None) -> None:

        self.macd_signal = macd_signal

        super().__init__(
            name=name or f'{macd_signal.name}_Histogram',
            features=[self.macd_signal],
            calculator=lambda f: (
                    self.macd_signal.macd.result - self.macd_signal.result
            )
        )

class LiquiditySpikes(Feature):

    def __init__(
            self,
            volume: Column,
            span: int = 20,
            z_score_threshold: int = 2,
            gradual: bool = False,
            name: str = None
    ) -> None:

        self.volume = volume
        self.span = span
        self.z_score_threshold = z_score_threshold

        super().__init__(
            name=name or (
                f'{volume.name}_'
                f'{'Gradual_' if gradual else ''}'
                f'Liquidity_Spikes_'
                f'{self.span}_{self.z_score_threshold}'
            ),
            features=[self.volume],
            calculator=lambda f: (
                liquidity_spikes(
                    self.volume.result,
                    gradual=gradual,
                    z_score_threshold=self.z_score_threshold,
                    span=span
                )
            )
        )

class ATR(Feature):

    def __init__(
            self, high: Column, low: Column, close: Column, name: str = None) -> None:

        self.high = high
        self.low = low
        self.close = close

        super().__init__(
            name=name or f'ATR',
            features=[self.high, self.low, self.close],
            calculator=lambda f: average_true_range(
                high=self.high.result,
                low=self.low.result,
                close=self.close.result
            )
        )

class SuperTrend(Feature):

    def __init__(
            self,
            feature: Feature,
            atr: ATR,
            span: int,
            factor: int,
            name: str = None
    ) -> None:

        self.feature = feature
        self.atr = atr
        self.span = span
        self.factor = factor

        super().__init__(
            name=name or f'{self.feature.name}_Super_Trend_{self.span}_{self.factor}',
            features=[self.atr, self.feature],
            calculator=lambda f: super_trend(
                data=self.feature.result, atr=self.atr.result,
                span=self.span, factor=self.factor
            )
        )
