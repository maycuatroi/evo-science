import typing

import pandas as pd
from tabulate import tabulate

from evo_science import FeatureSet
from evo_science.entities.metrics import BaseMetric
from evo_science.metric_lib import MetricLib


class BaseModel:
    def __init__(self, **kwargs):
        self.model = None

    def fit(self, x: FeatureSet, y: FeatureSet):
        """
        Fit the model to the data.

        Args:
            x (FeatureSet): Feature set for training data.
            y (FeatureSet): Target set for training data.
        """
        assert x._is_built and y._is_built, "FeatureSet is not built yet."
        x_data = x.to_numpy(is_train=True)
        y_data = y.to_numpy(is_train=True)
        self.model.fit(x_data, y_data)

    def predict(self, X):
        """
        Predict using the fitted model.

        Args:
            X (array-like): Features data for prediction.

        Returns:
            array-like: Predicted target values.
        """
        return self.model.predict(X)

    def evaluate(
        self,
        x: FeatureSet,
        y: FeatureSet,
        size: float = None,
        metrics: typing.List["str | BaseMetric | type"] = None,
        **kwargs,
    ):
        """
        Evaluate the model performance.

        Args:
            x (FeatureSet): Feature set for test data.
            y (FeatureSet): Target set for test data.
            size (float, optional): Size of the data to be evaluated. The fraction of the total data used.

        Returns:
            float: Mean Squared Error of the model predictions.
        """
        table_records = []
        for i, metric in enumerate(metrics):
            if isinstance(metric, str):
                metric: type = MetricLib.get_metric(metric)
                metrics[i] = metric(model=self, feature_set=x, target_feature=y)
            elif isinstance(metric, type):
                metrics[i] = metric(model=self, feature_set=x, target_feature=y)
            elif issubclass(metric.__class__, BaseMetric):
                metric.model = self
                metric.feature_set = x
                metric.target_feature = y
            else:
                assert issubclass(
                    metric.__class__, BaseMetric
                ), f"{metric.__class__.__name__} is not a subclass of BaseMetric"

        for metric in metrics:
            metric_value = metric.calculate()
            table_records.append([metric.name, metric_value])

        table = tabulate(
            tabular_data=table_records, headers=["Metric", "Value"], tablefmt="orgtbl",
        )
        print(table)

    def calculate_coefficients(self, x: FeatureSet) -> pd.DataFrame:
        """
        Get the coefficients of the features x to the target y.
        """
        df_cof = self._calculate_coefficients(x)

        table = tabulate(
            tabular_data=df_cof.values, headers=df_cof.columns, tablefmt="orgtbl"
        )
        print(table)
        return df_cof

    def plot(self, x: FeatureSet, y: FeatureSet, **kwargs):
        """
        Visualize the model predictions.

        Args:
            x (FeatureSet): Feature set for test data.
            y (FeatureSet): Target set for test data.
        """
        raise NotImplementedError

    def _calculate_coefficients(self, x):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _calculate_coefficients method."
        )
