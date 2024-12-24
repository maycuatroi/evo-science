import torch
import numpy as np
from typing import Union, Tuple
from evo_science.entities.metrics.base_metric import BaseMetric


class AP(BaseMetric):
    """
    Average Precision (AP)
    Compute the AP for a given set of iou thresholds.
    """

    def __init__(self, iou_thresholds: Union[torch.Tensor, np.ndarray], **kwargs):
        super().__init__(**kwargs)
        if isinstance(iou_thresholds, np.ndarray):
            self.iou_thresholds = torch.as_tensor(iou_thresholds)
        elif isinstance(iou_thresholds, torch.Tensor):
            self.iou_thresholds = iou_thresholds
        else:
            raise ValueError("`iou_thresholds` must be a numpy array or a torch tensor")

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AP for numpy inputs."""
        return self._calculate_torch(torch.from_numpy(y_true), torch.from_numpy(y_pred))

    def _calculate_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Main calculation method using PyTorch tensors."""
        # Calculate IoU and determine correct predictions
        iou = self._calculate_iou(y_true[:, 1:], y_pred[:, :4])
        correct = self._determine_correct_predictions(iou, y_true[:, 0:1], y_pred[:, 5])

        return self._compute_ap(
            tp=correct.cpu().numpy(),
            conf=y_pred[:, 4].cpu().numpy(),
            pred_cls=y_pred[:, 5].cpu().numpy(),
            target_cls=y_true[:, 0].cpu().numpy(),
        )

    def _calculate_iou(self, true_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
        """Calculate Intersection over Union between true and predicted boxes."""
        true_boxes = true_boxes.unsqueeze(1)
        pred_boxes = pred_boxes.unsqueeze(0)

        # Calculate intersection coordinates
        top_left = torch.max(true_boxes[..., :2], pred_boxes[..., :2])
        bottom_right = torch.min(true_boxes[..., 2:], pred_boxes[..., 2:])

        # Calculate areas
        intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
        true_area = (true_boxes[..., 2:] - true_boxes[..., :2]).prod(dim=2)
        pred_area = (pred_boxes[..., 2:] - pred_boxes[..., :2]).prod(dim=2)

        return intersection / (true_area + pred_area - intersection + 1e-7)

    def _determine_correct_predictions(
        self, iou: torch.Tensor, true_classes: torch.Tensor, pred_classes: torch.Tensor
    ) -> torch.Tensor:
        """Determine correct predictions based on IoU thresholds and class predictions."""
        correct = torch.zeros(
            (pred_classes.shape[0], self.iou_thresholds.shape[0]), dtype=torch.bool, device=pred_classes.device
        )

        for i, iou_threshold in enumerate(self.iou_thresholds):
            matches = torch.nonzero((iou >= iou_threshold) & (true_classes == pred_classes), as_tuple=False)
            if matches.shape[0]:
                true_box_indices = matches[:, 0]
                pred_box_indices = matches[:, 1]

                iou_scores = iou[true_box_indices, pred_box_indices].unsqueeze(1)
                matches = torch.cat((matches, iou_scores), 1)

                iou_sorted_indices = matches[:, 2].argsort(descending=True)
                matches = matches[iou_sorted_indices]

                unique_pred_indices = torch.unique(matches[:, 1], return_index=True)[1]
                matches = matches[unique_pred_indices]

                unique_true_indices = torch.unique(matches[:, 0], return_index=True)[1]
                matches = matches[unique_true_indices]

                matched_pred_indices = matches[:, 1].long()
                correct[matched_pred_indices, i] = True

        return correct

    def _compute_ap(self, tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray) -> float:
        """Compute Average Precision across classes and IoU thresholds."""
        # Sort by confidence
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]

        # Compute AP for each class and IoU threshold
        ap = np.zeros((nc, tp.shape[1]))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            if i.sum() == 0 or nt[ci] == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Calculate precision and recall
            recall = tpc / (nt[ci] + 1e-16)
            precision = tpc / (tpc + fpc)

            # Calculate AP for each IoU threshold
            for j in range(tp.shape[1]):
                ap[ci, j] = self._compute_ap_single(recall[:, j], precision[:, j])

        return ap.mean()

    @staticmethod
    def _compute_ap_single(recall: np.ndarray, precision: np.ndarray) -> float:
        """Compute AP for a single class and IoU threshold using 101-point interpolation."""
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute maximum precision for recall >= current recall
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Use 101-point interpolation (COCO method)
        x = np.linspace(0, 1, 101)
        return np.trapz(np.interp(x, mrec, mpre), x)
