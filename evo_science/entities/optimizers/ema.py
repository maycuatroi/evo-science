import torch
from torch import nn
import copy
from typing import Callable


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) implementation.

    Maintains a moving average of the model's parameters and buffers.
    Reference:
    - https://github.com/rwightman/pytorch-image-models
    - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000, updates: int = 0):
        self.ema_model = copy.deepcopy(model).eval()
        self.update_count = updates
        self.decay_fn = self._create_decay_function(decay, tau)
        self._freeze_ema_params()

    def _create_decay_function(self, decay: float, tau: float) -> Callable[[int], float]:
        return lambda x: decay * (1 - torch.exp(torch.tensor(-x / tau)).item())

    def _freeze_ema_params(self):
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model: nn.Module):
        if hasattr(model, "module"):
            model = model.module

        with torch.no_grad():
            self.update_count += 1
            current_decay = self.decay_fn(self.update_count)

            for ema_param, model_param in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
                if ema_param.dtype.is_floating_point:
                    ema_param.mul_(current_decay).add_(model_param.detach(), alpha=1 - current_decay)
