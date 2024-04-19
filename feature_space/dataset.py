# dataset.py

from dataclasses import dataclass, field

import pandas as pd

from feature_space.features import Feature

__all__ = [
    "Dataset"
]

@dataclass
class Dataset:

    name: str = 'Dataset'
    features: list[Feature] = field(default_factory=list)
    datasets: list['Dataset'] = field(default_factory=list)

    def __hash__(self) -> int:

        return hash(self.name)

    @property
    def all_features_names(self) -> list[str]:

        return list(set(self.datasets_features_names + self.features_names))

    @property
    def features_names(self) -> list[str]:

        return [f.name for f in self.features]

    @property
    def datasets_names(self) -> list[str]:

        return [d.name for d in self.datasets]

    @property
    def datasets_features_names(self) -> list[str]:

        names = set()

        for dataset in self.datasets:
            names.update(dataset.all_features_names)

        return list(names)

    @property
    def datasets_features(self) -> list[Feature]:

        features = set()

        for dataset in self.datasets:
            features.update(dataset.all_features)

        return list(features)

    @property
    def all_features(self) -> list[Feature]:

        return list(set(self.datasets_features + self.features_names))

    @property
    def results(self) -> list[pd.Series]:

        if not self.calculated:
            raise RuntimeError('Not all features are calculated.')

        return [f.result for f in set(self.features)]

    @property
    def features_calculated(self) -> bool:

        return all(f.calculated for f in set(self.features))

    @property
    def datasets_calculated(self) -> bool:

        return all(d.calculated for d in set(self.datasets))

    @property
    def calculated(self) -> bool:

        return self.features_calculated and self.datasets_calculated

    def calculate_features(
            self,
            data: pd.DataFrame,
            cached: bool = True,
            override: bool = False
    ) -> 'Dataset':

        for feature in self.features:
            feature.calculate(data=data, cached=cached, override=override)

        return self

    def calculate_datasets(
            self,
            data: pd.DataFrame,
            cached: bool = True,
            override: bool = False
    ) -> 'Dataset':

        for dataset in self.datasets:
            dataset.calculate(data=data, cached=cached, override=override)

        return self

    def calculate(
            self,
            data: pd.DataFrame,
            cached: bool = True,
            override: bool = False
    ) -> 'Dataset':

        self.calculate_datasets(data=data, cached=cached, override=override)
        self.calculate_features(data=data, cached=cached, override=override)

        return self
