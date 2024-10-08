from torch import nn
import torch
from torch.nn import functional as F
from evo_science.entities.metrics.iou import IOU


class BoxLoss(nn.Module):
    def __init__(self, dfl_ch):
        super().__init__()
        self.dfl_ch = dfl_ch

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        loss_iou = self._compute_iou_loss(pred_bboxes, target_bboxes, target_scores, target_scores_sum, fg_mask)
        loss_dfl = self._compute_dfl_loss(
            pred_dist, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        return loss_iou, loss_dfl

    def _compute_iou_loss(self, pred_bboxes, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = IOU.compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        return ((1.0 - iou) * weight).sum() / target_scores_sum

    def _compute_dfl_loss(self, pred_dist, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        target = self._prepare_dfl_target(anchor_points, target_bboxes)
        loss_dfl = self._distribution_focal_loss(pred_dist[fg_mask].view(-1, self.dfl_ch + 1), target[fg_mask])
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        return (loss_dfl * weight).sum() / target_scores_sum

    def _prepare_dfl_target(self, anchor_points, target_bboxes):
        a, b = target_bboxes.chunk(2, -1)
        target = torch.cat((anchor_points - a, b - anchor_points), -1)
        return target.clamp(0, self.dfl_ch - 0.01)

    @staticmethod
    def _distribution_focal_loss(pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        left_loss = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        right_loss = F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)
