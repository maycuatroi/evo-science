import torch.nn as nn

from evo_science.entities.models.abstract_model import AbstractModel


class AbstractTorchModel(nn.Module, AbstractModel):
    pass
