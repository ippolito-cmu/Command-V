import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn as nn


class StrEnum(str, Enum):
    def __str__(self):
        return self.value

    @classmethod
    def snake_case_name(cls):
        return ''.join(['_' + c.lower() if c.isupper() else c for c in cls.__name__]).lstrip('_')


class NumpyTorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, bool):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.dtype) or isinstance(obj, torch.dtype):
            return str(obj)
        elif torch.is_tensor(obj):
            if obj.numel() == 1:
                return obj.item()
            else:
                return obj.detach().cpu().numpy().tolist()
        return super(NumpyTorchEncoder, self).default(obj)


class ActivationPoint(StrEnum):
    LAYER_OUT = 'Layer.out'  # Residual
    MLP_IN = 'MLP.in'  # Post attention norm, pre MLP
    MLP_POST_ACT_FN = 'MLP.post'  # post SiLU/SwiGLU activation
    MLP_OUT = 'MLP.out'  # post SiLU/etc activation


class BaseConverter(ABC, nn.Module):
    """Abstract base class for all converters."""

    def __init__(self, input_dim: int, output_dim: int, device=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss = None
        if device is None:
            # Import here to avoid circular imports
            from .activations import get_device
            device = get_device()
        self.device = device

    @abstractmethod
    def forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def backward_transform(self, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        pass

    def compute_loss_and_metrics(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes loss and metrics in one go."""
        y_pred = self.forward_transform(x)
        forward_loss = torch.mean((y_pred - y) ** 2)
        backward_loss = torch.mean((self.backward_transform(y_pred) - x) ** 2)  # Use y_pred for cycle consistency
        x_cycle_loss = torch.mean((self.backward_transform(self.forward_transform(x)) - x) ** 2)
        y_cycle_loss = torch.mean((self.forward_transform(self.backward_transform(y)) - y) ** 2)
        weighted_loss = forward_loss + x_cycle_loss + 0.1 * (backward_loss + y_cycle_loss)

        forward_corr = torch.mean(torch.cosine_similarity(y_pred, y, dim=-1)).item()
        backward_corr = torch.mean(torch.cosine_similarity(self.backward_transform(y), x, dim=-1)).item()
        loss = weighted_loss, {
            'forward_loss': forward_loss.item(),
            'backward_loss': backward_loss.item(),
            'x_cycle_loss': x_cycle_loss.item(),
            'y_cycle_loss': y_cycle_loss.item(),
            'forward_mse': forward_loss.item(),
            'backward_mse': backward_loss.item(),
            'cycle_mse_x': x_cycle_loss.item(),
            'cycle_mse_y': y_cycle_loss.item(),
            'forward_correlation': forward_corr,
            'backward_correlation': backward_corr
        }
        self.loss = loss
        return loss

    def save(self, filepath: str):
        """Saves the entire converter object."""
        torch.save(self, filepath + ".pth")

    @classmethod
    def load(cls, filepath: str) -> 'BaseConverter':
        """Loads a converter object."""
        return torch.load(filepath + ".pth", weights_only=False)


class ReconstructedIntervention(nn.Module):
    """Base class for reconstructed interventions."""
    pass


class ReconstructedNodireftIntervention(ReconstructedIntervention):
    """Reconstructed NodiReFT intervention for behavioral transfer."""
    
    def __init__(self, embed_dim: int, interchange_dim: int, low_rank_dimension: int,
                 state_dict: Dict[str, torch.Tensor] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.interchange_dim = interchange_dim
        self.low_rank_dimension = low_rank_dimension
        self.proj_layer = nn.Linear(embed_dim, low_rank_dimension, bias=True)
        self.learned_source = nn.Linear(embed_dim, low_rank_dimension, bias=True)

        # Load the state dict if provided
        if state_dict:
            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply intervention to input activations."""
        return torch.matmul(
            (self.learned_source(x)), self.proj_layer.weight
        )
