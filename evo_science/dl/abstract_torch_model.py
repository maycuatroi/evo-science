import torch
import torch.nn as nn

from evo_science.entities.models.abstract_model import AbstractModel


class AbstractTorchModel(nn.Module, AbstractModel):

    def load_weight(self, checkpoint_path: str):
        """
        Load weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            self: The model instance with loaded weights.
        """
        # Load the current model state
        model_state = self.state_dict()

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_state = checkpoint["model"].float().state_dict()

        # Filter and load matching weights
        compatible_weights = {
            k: v for k, v in checkpoint_state.items() if k in model_state and v.shape == model_state[k].shape
        }

        # Update the model with compatible weights
        self.load_state_dict(compatible_weights, strict=False)

        return self

    def get_criterion(self):
        raise NotImplementedError("This method must be implemented in the subclass.")

    def clip_gradients(self, max_norm=10.0):
        """
        Clip gradients of the model's parameters.

        Args:
            max_norm (float): The maximum norm value for gradient clipping. Default is 10.0.
        """
        parameters = self.parameters()
        nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
