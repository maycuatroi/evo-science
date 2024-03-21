import typing

import numpy as np
import pandas as pd
from tabulate import tabulate

from evo_science import BaseFeature
from evo_science.entities.evs_logger import EvoScienceLogger


class FeatureSet:
    def __init__(
        self,
        features: typing.List["BaseFeature | type"],
        auto_skip_nan: bool = True,
        parents=None,
    ):
        self.auto_skip_nan = auto_skip_nan
        for i, feature in enumerate(features):
            if isinstance(feature, type):
                features[i] = feature()

        self._features = features
        self.logger = EvoScienceLogger()
        self._is_built = False
        self.__parents = parents
        if not parents:  # if no parents, then it is the root
            table = tabulate(
                [[f.column_name, f.column_type, f.categories] for f in features],
                headers=["Feature", "Type", "Categories"],
                tablefmt="grid",
            )
            self.logger.info(
                f"FeatureSet is built with the following features:\n{table}"
            )

    @property
    def column_names(self):
        return [x.column_name for x in self.features]

    @property
    def hard_column_names(self):
        column_names = []
        for feature in self.features:
            if feature.is_skip:
                continue
            if feature.virtual:
                column_names += feature.use_columns
            else:
                column_names.append(feature.column_name)

    @property
    def features(self):
        return [x for x in self._features if not x.is_skip]

    def clean(self, df):
        for feature in self.features:
            df = feature.clean(df=df)
        return df

    def encode(self, df):
        for feature in self.features:
            if feature.is_skip:
                continue
            df.loc[:, feature.column_name] = feature.encode(
                df[feature.column_name].values
            )
        return df

    def build(self, csv_path=None, df=None, train_test_split=None, seed=42):
        assert csv_path or df, "Either csv_path or df should be provided."
        assert not (csv_path and df), "Only one of csv_path or df should be provided."

        df = pd.read_csv(csv_path) if csv_path else df

        df = self.build_virtual_feature(df)  # for virtual features

        if self.auto_skip_nan:
            for i, feature in enumerate(self.features):
                try:
                    df_feature = df[feature.column_name]
                except KeyError as e:
                    self.logger.error(
                        f"Feature {feature.column_name} is not found in the data. {e}\n valid columns are {df.columns}"
                    )
                    raise e
                is_all_nan = np.all(pd.isna(df_feature))
                if is_all_nan:
                    self.features[i].is_skip = True
                    self.logger.warning(
                        f"Feature {feature.column_name} is skipped because it contains only NaN values."
                    )

        df = self.clean(df)
        df = self.fillna(df)
        df = self.encode(df)

        if train_test_split:
            train_size = int(len(df) * (1 - train_test_split))
            df_train = df.sample(n=train_size, random_state=seed)
            df_test = df.drop(df_train.index)
            for i, feature in enumerate(self.features):
                self.features[i].train_data = df_train[feature.column_name].values
                self.features[i].test_data = df_test[feature.column_name].values
            return df_train, df_test
        else:
            for i, feature in enumerate(self.features):
                self.features[i].train_data = df[feature.column_name].values
                self.features[i].test_data = df[feature.column_name].values
        self._is_built = True
        if self.__parents:
            for parent in self.__parents:
                parent._is_built = True
        return df

    def __add__(self, other):
        return FeatureSet(self.features + other.features, parents=[self, other])

    def to_numpy(self, is_train=True):
        # Adding dtype=object, for handling inhomogeneous data
        data = [
            feature.to_numpy(is_train=is_train)
            for feature in self.features
            if not feature.is_skip
        ]
        data = np.array(data).T
        return data

    def fillna(self, df):
        for feature in self.features:
            df[feature.column_name] = feature.fillna(df[feature.column_name])
        return df

    @property
    def df(self):
        data = self.to_numpy()
        column_names = [x.column_name for x in self.features if not x.is_skip]
        df = pd.DataFrame(data=data, columns=column_names)
        return df

    def build_virtual_feature(self, df):
        for feature in self.features:
            if feature.virtual:
                df[feature.column_name] = feature.compute(df)
        return df
