import typing

import numpy as np

if typing.TYPE_CHECKING:
    from evo_science import BaseModel, FeatureSet, BaseFeature


class BaseMetric:
    name = "base_metric"

    def __init__(
        self,
        model: "BaseModel" = None,
        feature_set: "FeatureSet" = None,
        target_feature: "BaseFeature" = None,
        **kwargs
    ):
        self.feature_set = feature_set
        self.target_feature = target_feature
        self.model = model
        self._on_init(**kwargs)

    def _calculate_np(self, y_true: np.array, y_pred: np.array):
        """
        Calculate the metric using numpy arrays.
        Args:
            y_true (np.array): True target values.
            y_pred (np.array): Predicted target values.
        Returns:
            float: Metric value.
        """
        raise NotImplementedError

    def calculate(self):
        """
        Calculate the slope of the model predictions.

        Returns:
            float: Slope of the model predictions.
        """
        y_true_np = self.target_feature.to_numpy(is_train=False)
        y_pred_np = self.model.predict(self.feature_set.to_numpy(is_train=False))

        return self._calculate_np(y_true_np, y_pred_np)

    def _on_call(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self._on_call()
        return self

    def _on_init(self):
        pass
