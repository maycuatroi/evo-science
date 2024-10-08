import torch
import numpy as np
from evo_science.entities.metrics.base_metric import BaseMetric


class AP(BaseMetric):
    """
    Average Precision (AP)
    Compute the AP for a given set of iou thresholds.
    """

    def __init__(self, iou_thresholds: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.iou_thresholds = iou_thresholds

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        return self._calculate_torch(torch.from_numpy(y_true), torch.from_numpy(y_pred))

    def _calculate_np(self, y_true: np.ndarray, y_pred: np.ndarray):
        return self._calculate_torch(torch.from_numpy(y_true), torch.from_numpy(y_pred))

    def _calculate_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        iou = self._calculate_iou(y_true[:, 1:], y_pred[:, :4])
        correct = self._determine_correct_predictions(iou, y_true[:, 0:1], y_pred[:, 5])
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        return self._compute_ap(correct.cpu().numpy(), y_pred[:, 4], y_pred[:, 5], y_true[:, 0])

    def _calculate_iou(self, true_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
        true_boxes = true_boxes.unsqueeze(1)
        pred_boxes = pred_boxes.unsqueeze(0)

        lt = torch.max(true_boxes[..., :2], pred_boxes[..., :2])
        rb = torch.min(true_boxes[..., 2:], pred_boxes[..., 2:])

        intersection = (rb - lt).clamp(min=0).prod(dim=2)
        true_area = (true_boxes[..., 2:] - true_boxes[..., :2]).prod(dim=2)
        pred_area = (pred_boxes[..., 2:] - pred_boxes[..., :2]).prod(dim=2)

        return intersection / (true_area + pred_area - intersection + 1e-7)

    def _determine_correct_predictions(
        self, iou: torch.Tensor, true_classes: torch.Tensor, pred_classes: torch.Tensor
    ) -> torch.Tensor:
        correct = torch.zeros(
            (pred_classes.shape[0], self.iou_thresholds.shape[0]), dtype=torch.bool, device=pred_classes.device
        )

        for i, iou_threshold in enumerate(self.iou_thresholds):
            matches = torch.nonzero((iou >= iou_threshold) & (true_classes == pred_classes), as_tuple=False)
            if matches.shape[0]:
                matches = torch.cat((matches, iou[matches[:, 0], matches[:, 1]].unsqueeze(1)), 1)
                matches = matches[matches[:, 2].argsort(descending=True)]
                matches = matches[torch.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[torch.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].long(), i] = True

        return correct

    def _compute_ap(self, tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray):
        # Sort by confidence
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes

        # Compute AP for each class and IoU threshold
        ap = np.zeros((nc, tp.shape[1]))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            if i.sum() == 0 or nt[ci] == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (nt[ci] + 1e-16)

            # Precision
            precision = tpc / (tpc + fpc)

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = self._compute_ap_single(recall[:, j], precision[:, j])

        return ap.mean()

    def _compute_ap_single(self, recall: np.ndarray, precision: np.ndarray) -> float:
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

        return ap
