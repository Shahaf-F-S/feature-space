# dataset.py

from dataclasses import dataclass, field

import pandas as pd

from feature_space.feature import Feature

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

        return [f.name for f in self.all_features]

    @property
    def features_names(self) -> list[str]:

        return [f.name for f in self.features]

    @property
    def datasets_names(self) -> list[str]:

        return [d.name for d in self.datasets]

    @property
    def datasets_features_names(self) -> list[str]:

        return [f.name for f in self.datasets_features]

    @property
    def datasets_features(self) -> list[Feature]:

        features = []

        for dataset in self.datasets:
            for feature in dataset.all_features:
                if feature not in features:
                    features.append(feature)

        return features

    @property
    def all_features(self) -> list[Feature]:

        features = self.features.copy()

        for feature in self.datasets_features:
            if feature not in features:
                features.append(feature)

        return features

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

    def clear_features(self) -> None:

        for feature in self.features:
            feature.clear()

    def clear_datasets(self) -> None:

        for dataset in self.datasets:
            dataset.clear()

    def clear(self) -> None:

        self.clear_datasets()
        self.clear_features()
