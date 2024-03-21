import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from evo_science.entities.metrics.base_binary_classify_metric import (
    BaseBinaryClassifyMetric,
)


class RocCurve(BaseBinaryClassifyMetric):
    name = "Roc Curve"

    def _on_init(self, threshold=0.5, plot=False, **kwargs):
        self.threshold = threshold
        self.plot = plot

    def _calculate_np(self, y_true: np.array, y_pred: np.array):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        if self.plot:
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.show()

        return fpr.mean(), tpr.mean(), thresholds.mean()
