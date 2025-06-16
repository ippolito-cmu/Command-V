import importlib
from typing import Tuple

import torch
from safetensors import safe_open
from transformers import AutoConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif importlib.util.find_spec('torch_xla') is not None:
        import torch_xla.core.xla_model as xm
        try:
            if xm.get_xla_supported_devices(devkind='TPU'):
                return xm.xla_device()
        except RuntimeError:
            pass

    return torch.device('cpu')


def get_layer_name(i):
    return f"model.layers.{i}"


def get_layer_keys(activation_file_path):
    with safe_open(activation_file_path, framework="pt", device="cpu") as f:
        return [key for key in f.keys() if key.startswith('model.layers')]


def get_num_layers(activation_file_path):
    return len(get_layer_keys(activation_file_path))


def get_model_layer_counts(model_or_model_id):
    if isinstance(model_or_model_id, str):
        config = AutoConfig.from_pretrained(model_or_model_id)
        if hasattr(config, 'num_hidden_layers'):
            return config.num_hidden_layers
        elif hasattr(config, 'n_layers'):
            return config.n_layers
        elif hasattr(config, 'text_config'):
            raise NotImplementedError("Multimodal model are not supported yet.")

    else:
        if hasattr(model_or_model_id, 'config') and hasattr(model_or_model_id.config, 'num_hidden_layers'):
            return model_or_model_id.config.num_hidden_layers

        if hasattr(model_or_model_id, 'layers'):
            return len(model_or_model_id.layers)

        if hasattr(model_or_model_id, 'layer'):
            return len(model_or_model_id.layer)

        if hasattr(model_or_model_id, 'model'):
            return get_model_layer_counts(model_or_model_id.model)

    raise ValueError("Could not determine layer count for this model architecture")


def load_residual_pairs(file1: str, file2: str, layer1: int, layer2: int, activ_profile_lim: int = -1) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """Load residual outputs from two models for specific layers."""
    keys1 = get_layer_keys(file1)
    keys2 = get_layer_keys(file2)
    if layer1 >= len(keys1) or layer2 >= len(keys2):
        raise ValueError(f"Layer indices ({layer1}, {layer2}) exceed model dimensions ({len(keys1)}, {len(keys2)})")

    key1 = get_layer_name(layer1)
    key2 = get_layer_name(layer2)

    with safe_open(file1, framework="pt", device="cpu") as f1, \
            safe_open(file2, framework="pt", device="cpu") as f2:
        residual1 = f1.get_tensor(key1).float()
        residual2 = f2.get_tensor(key2).float()
        if activ_profile_lim is not None and activ_profile_lim > 0:
            residual1 = residual1[:activ_profile_lim]
            residual2 = residual2[:activ_profile_lim]
        return residual1, residual2
