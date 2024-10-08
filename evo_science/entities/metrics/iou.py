from evo_science.entities.metrics.base_metric import BaseMetric
import torch
import numpy as np


class IOU(BaseMetric):
    name = "Intersection over Union"

    def _calculate_np(self, y_true, y_pred):
        return np.mean(np.diag(y_true @ y_pred.T))

    @staticmethod
    def compute_iou(box1, box2, eps=1e-7):
        # Returns Complete Intersection over Union (CIoU) of box1(1,4) to box2(n,4)

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

        # Calculate width and height of boxes
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

        # Calculate intersection area
        inter = torch.clamp((torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)), min=0) * torch.clamp(
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)), min=0
        )

        # Calculate union area
        union = w1 * h1 + w2 * h2 - inter + eps

        # Calculate IoU
        iou = inter / union

        # Calculate the convex (smallest enclosing box) width and height
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        # Calculate diagonal distance
        c2 = cw.pow(2) + ch.pow(2) + eps

        # Calculate center distance
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4

        # Calculate aspect ratio consistency term
        v = (4 / (torch.pi**2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).pow(2)

        # Calculate alpha for CIoU
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))

        # Return CIoU
        return iou - (rho2 / c2 + v * alpha)
