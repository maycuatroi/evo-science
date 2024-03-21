import matplotlib.pyplot as plt
from sklearn import metrics

from evo_science.entities.metrics.base_binary_classify_metric import (
    BaseBinaryClassifyMetric,
)


class AUC(BaseBinaryClassifyMetric):
    name = "Area Under the Curve"

    def _on_init(self, threshold=0.5, plot=False, **kwargs):
        self.threshold = threshold
        self.plot = plot

    def _calculate_np(self, y_true, y_pred):
        auc = metrics.roc_auc_score(y_true, y_pred)
        if self.plot:
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            plt.figure()
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label="ROC curve (area = %0.2f)" % auc,
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic Curve")
            plt.legend(loc="lower right")
            plt.show()
        return auc
