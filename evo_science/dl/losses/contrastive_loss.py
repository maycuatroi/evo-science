import torch
import torch.nn as nn
import torch.nn.functional as F

from evo_science.dl.losses.base_loss import BaseLoss


class ContrastiveLoss(BaseLoss):
    """
    Contrastive loss function.

    Based on: Dimensionality Reduction by Learning an Invariant Mapping
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor):
        """
        Forward pass for contrastive loss.

        Args:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.
            y (torch.Tensor): Label tensor.

        Returns:
            torch.Tensor: Contrastive loss.
        """
        dist = F.pairwise_distance(x1, x2)
        loss = y * torch.pow(dist, 2) + (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return torch.mean(loss)
