# features.py

from typing import Callable, ParamSpec, ParamSpecKwargs
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = (
    'Feature',
    'Column'
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

_P = ParamSpec('_P')
P = ParamSpecKwargs(_P)

@dataclass
class Feature:

    name: str
    features: list['Feature'] = field(default_factory=list)
    kwargs: P = field(default_factory=dict)
    calculator: Callable[['Feature'], pd.Series] = field(default=None, repr=False)
    data: pd.DataFrame | None = field(default=None, repr=False)
    result: pd.Series | None = field(default=None, repr=False)

    def __hash__(self) -> int:

        return hash(self.name)

    @property
    def calculated(self) -> bool:

        return self.result is not None

    @property
    def all_features_names(self) -> list[str]:

        return list(set([self.name] + self.features_names))

    @property
    def features_names(self) -> list[str]:

        return [f.name for f in self.features]

    def calculate(
            self,
            data: pd.DataFrame,
            cached: bool = True,
            override: bool = False
    ) -> 'Feature':

        if cached and (self.result is not None):
            return self

        if (self.name in data.columns) and not override:
            self.result = data[self.name]

            return self

        for feature in self.features:
            feature.calculate(data, cached=cached)

        if self.calculator is None:
            raise ValueError(f'Feature calculator of {self} is not defined.')

        self.data = data

        data[self.name] = self.result = self.calculator(self)

        return self

    def clear(self) -> None:

        self.result = None
        self.data = None

class Column(Feature):

    def __init__(self, name: str) -> None:

        super().__init__(name=name, calculator=lambda f: f.data[self.name])
