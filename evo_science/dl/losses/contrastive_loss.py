import torch
import torch.nn as nn
import torch.nn.functional as F

from evo_science.dl.losses.base_loss import BaseLoss


class ContrastiveLoss(BaseLoss):
    """
    Contrastive loss function.
    
    Based on: Dimensionality Reduction by Learning an Invariant Mapping
    """
    def __init__(self, margin:float=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1:torch.Tensor, output2:torch.Tensor, label:torch.Tensor):
        """
        Forward pass for contrastive loss.

        Args:
            output1 (torch.Tensor): First output tensor.
            output2 (torch.Tensor): Second output tensor.
            label (torch.Tensor): Label tensor.

        Returns:
            torch.Tensor: Contrastive loss.
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive
