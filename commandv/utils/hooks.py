from typing import Dict, List, Tuple

import torch

from .types import ActivationPoint


class ActivationHook:
    def __init__(self, activation_point: ActivationPoint, name: str, dtype: torch.dtype):
        self.activation_point = activation_point
        self.name = name
        self.dtype = dtype
        self.activation = None

    def __call__(self, module, input_tensor, output):
        if self.activation_point == ActivationPoint.LAYER_OUT:
            self.activation = output if isinstance(output, torch.Tensor) else output[0]

        else:
            raise NotImplementedError(f"Activation point {self.activation_point} not implemented")

        self.activation = self.activation.to(self.dtype).cpu()


def hook_model_layers(
        model: torch.nn.Module,
        layer_names: List[str],
        activation_point: ActivationPoint,
        dtype: torch.dtype,
) -> Tuple[Dict[str, ActivationHook], List]:
    """
    Capture activations from specified layers.

    Args:
        model: The model to capture from
        layer_names: List of layer names
        activation_point: Which activation point to capture (from ACTIVATION_POINT)
        dtype: The desired dtype for the activations

    Returns:
        Tuple of (activations dict, hooks list)
    """
    removable_handles = []
    hook_instances = {}  # Store the actual hook instances

    for name in layer_names:
        layer = dict(model.named_modules())[name]
        hook = ActivationHook(activation_point=activation_point, name=name, dtype=dtype)
        hook_instances[name] = hook

        if activation_point == ActivationPoint.LAYER_OUT:
            removable_handles.append(layer.register_forward_hook(lambda m, i, o, h=hook: h(m, i, o)))
        else:
            raise NotImplementedError(f"Activation point {activation_point} not implemented")

    # Return a dictionary that dynamically collects activations from hooks

    return hook_instances, removable_handles


def remove_hooks(hooks: List):
    """Remove hooks from the model"""
    for hook in hooks:
        hook.remove()


def get_last_token_activations(activations: Dict[str, torch.Tensor],
                               last_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Extract activations for the last token of each sequence in batch.

    Args:
        activations: Either a dict mapping to a tensor of shape [batch_size, seq_length, hidden_dim]
        last_indices: 1D tensor of shape [batch_size] containing last token positions to extract
    """
    assert isinstance(activations, dict)
    # Handle dictionary case (multiple tensors)
    last_token = {}
    for name, tensor in activations.items():
        assert len(tensor.shape) == 3
        if len(last_indices) == 1:
            if last_indices[0] + 1 != tensor.shape[1]:
                print("WARN: mismatch between last index and shape @batchsize=1", last_indices[0] + 1, "!=",
                      tensor.shape[1])
        last_token[name] = tensor[torch.arange(tensor.size(0), device=tensor.device), last_indices.cpu()]
    return last_token
