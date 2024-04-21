# features.py

import dill
from uuid import uuid4
from typing import Callable, ParamSpec, ParamSpecKwargs
from dataclasses import dataclass, field

import pandas as pd

__all__ = (
    'Feature',
    'Column'
)

_P = ParamSpec('_P')
P = ParamSpecKwargs(_P)

@dataclass
class Feature:

    name: str
    id: str = field(default_factory=lambda: str(uuid4()))
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

        names = self.features_names.copy()

        if self.name not in names:
            names.insert(0, self.name)

        return names

    @property
    def all_features(self) -> set['Feature']:

        features = set()

        for feature in self.features:
            features.update(feature.all_features)

        features.add(self)

        return features

    @property
    def features_names(self) -> list[str]:

        return [f.name for f in self.features]

    def copy(self) -> "Feature":

        data = self.__reduce__()

        copy = data[0](type(self), data[1][0], data[-1])

        return copy

    def save(self, path: str) -> None:

        copy = self.copy()
        copy.clear()

        with open(path, 'wb') as file:
            dill.dump(self, file)

    @classmethod
    def load(cls, path: str) -> "Feature":

        with open(path, 'rb') as file:
            return dill.load(file)

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
