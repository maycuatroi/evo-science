from evo_science.dl.losses.base_loss import BaseLoss

import numpy as np


class TripletLossEuclidean(BaseLoss):

    def __init__(self, margin=0.5):
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """Forward pass for triplet loss.

        Args:
            anchor (np.ndarray): Anchor embeddings.
            positive (np.ndarray): Positive embeddings.
            negative (np.ndarray): Negative embeddings.

        Returns:
            float: Mean triplet loss.
        """
        pos_dist = np.linalg.norm(anchor - positive, axis=1)
        neg_dist = np.linalg.norm(anchor - negative, axis=1)
        losses = np.maximum(0, pos_dist - neg_dist + self.margin)
        return np.mean(losses)
