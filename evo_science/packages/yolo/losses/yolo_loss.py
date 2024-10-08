from torch import nn
import torch
from evo_science.packages.yolo.losses.assigner import Assigner
from evo_science.packages.yolo.losses.box_loss import BoxLoss
from evo_science.packages.yolo.utils import make_anchors


class YoloLoss:
    def __init__(
        self,
        num_classes,
        num_outputs,
        num_channels,
        stride,
        box_gain=7.5,
        cls_gain=0.5,
        dfl_gain=1.5,
        top_k=10,
        alpha=0.5,
        beta=6.0,
    ):
        """
        Initialize the YoloLoss.

        Args:
            num_classes (int): Number of classes.
            num_outputs (int): Number of outputs per anchor.
            num_channels (int): Number of channels for distribution focal loss.
            stride (list): Stride of the model's output.
            box_gain (float): Box loss gain.
            cls_gain (float): Class loss gain.
            dfl_gain (float): Distribution focal loss gain.
            top_k (int): Top k predictions to consider for loss calculation.
            alpha (float): Alpha parameter for the assigner.
            beta (float): Beta parameter for the assigner.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loss computed on device: {device}")

        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.stride = stride
        self.nc = num_classes
        self.no = num_outputs
        self.reg_max = num_channels
        self.device = device

        self.box_loss = BoxLoss(num_channels - 1).to(device)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = Assigner(top_k=top_k, nc=self.nc, alpha=alpha, beta=beta)

        self.project = torch.arange(num_channels, dtype=torch.float, device=device)

    def box_decode(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4)
        pred_dist = pred_dist.softmax(3)
        pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        return torch.cat((x1y1, x2y2), -1)

    def __call__(self, outputs, targets):
        loss_cls = torch.zeros(1, device=self.device)
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)

        x = torch.cat([i.view(outputs[0].shape[0], self.no, -1) for i in outputs], 2)
        pred_distri, pred_scores = x.split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        data_type = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        input_size = torch.tensor(outputs[0].shape[2:], device=self.device, dtype=data_type) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(outputs, self.stride, offset=0.5)

        idx = targets["idx"].view(-1, 1)
        cls = targets["cls"].view(-1, 1)
        box = targets["box"]

        targets = torch.cat((idx, cls, box), dim=1).to(self.device)
        if targets.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
            x = gt[..., 1:5].mul_(input_size[[1, 0, 1, 0]])
            y = torch.empty_like(x)
            dw = x[..., 2] / 2  # half-width
            dh = x[..., 3] / 2  # half-height
            y[..., 0] = x[..., 0] - dw  # top left x
            y[..., 1] = x[..., 1] - dh  # top left y
            y[..., 2] = x[..., 0] + dw  # bottom right x
            y[..., 3] = x[..., 1] + dh  # bottom right y
            gt[..., 1:5] = y
        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.box_decode(anchor_points, pred_distri)
        assigned_targets = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_bboxes, target_scores, fg_mask = assigned_targets

        target_scores_sum = max(target_scores.sum(), 1)

        loss_cls = self.cls_loss(pred_scores, target_scores.to(data_type)).sum() / target_scores_sum  # BCE

        # Box loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss_box, loss_dfl = self.box_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss_box *= self.box_gain  # box gain
        loss_cls *= self.cls_gain  # cls gain
        loss_dfl *= self.dfl_gain  # dfl gain

        return loss_box, loss_cls, loss_dfl
