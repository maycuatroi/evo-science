import torch
import numpy as np

from evo_science.entities.metrics.ap import AP


class MAP(AP):
    """
    Mean Average Precision (mAP)
    Compute the mAP for a given set of iou thresholds.
    Formula:
        AP = 1/K * sum(precision_k)
        mAP = 1/N * sum(AP)
    """

    def __init__(self, iou_thresholds: torch.Tensor, **kwargs):
        super().__init__(iou_thresholds=iou_thresholds, **kwargs)

    def _calculate_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        iou = self._calculate_iou(y_true[:, 1:], y_pred[:, :4])
        correct = self._determine_correct_predictions(iou, y_true[:, 0:1], y_pred[:, 5])
        return correct
