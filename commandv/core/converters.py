"""
Command-V converter implementation.
Provides clean pseudoinverse-based converters for activation space mapping.
"""

import gzip
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn

from ..utils.activations import load_residual_pairs, get_device, get_num_layers
from ..utils.paths import get_model_name, get_activation_path
from ..utils.types import BaseConverter


class PinvConverter(BaseConverter):
    """Pseudoinverse converter - core Command-V component with compression support."""

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.forward_matrix = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=False)
        self.backward_matrix = nn.Parameter(torch.zeros(output_dim, input_dim), requires_grad=False)

    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        self.forward_matrix.data = torch.linalg.pinv(x) @ y
        self.backward_matrix.data = torch.linalg.pinv(y) @ x

    def forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.forward_matrix.device != x.device:
            self.forward_matrix = self.forward_matrix.to(x.device)
        return x @ self.forward_matrix

    def backward_transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.backward_matrix.device != y.device:
            self.backward_matrix = self.backward_matrix.to(y.device)
        return y @ self.backward_matrix

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        y_pred = self.forward_transform(x)
        return torch.mean((y_pred - y) ** 2).item()

    def save_compressed(self, filepath: str) -> None:
        """Save converter with compression to reduce file size."""
        converter_data = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'forward_matrix': self.forward_matrix.detach().cpu(),
            'backward_matrix': self.backward_matrix.detach().cpu(),
            'loss': getattr(self, 'loss', None)
        }

        with gzip.open(filepath + '.pkl.gz', 'wb') as f:
            pickle.dump(converter_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.save(filepath)

    @classmethod
    def load_compressed(cls, filepath: str) -> 'PinvConverter':
        """Load converter from compressed format."""
        compressed_path = filepath + '.pkl.gz'
        if Path(compressed_path).exists():
            # Load from compressed format
            with gzip.open(compressed_path, 'rb') as f:
                data = pickle.load(f)

            # Recreate converter
            converter = cls(
                input_dim=data['input_dim'],
                output_dim=data['output_dim'],
            )
            converter.forward_matrix.data = data['forward_matrix']
            converter.backward_matrix.data = data['backward_matrix']
            converter.loss = data.get('loss', None)

            return converter
        else:
            # Fall back to standard PyTorch format
            return cls.load(filepath)


def create_linear_layer_mapping(layers_source: List[int], layers_target: List[int],
                                n_model1_layers: int, n_model2_layers: int) -> Dict[int, int]:
    """Create linear mapping between layers based on relative positions."""
    if len(layers_target) == len(layers_source):
        return {l2: l1 for l1, l2 in zip(layers_source, layers_target)}

    scale = n_model2_layers / n_model1_layers
    return {l2: min(layers_target, key=lambda l: abs(l - scale * l2)) for l2 in layers_source}


def create_converter_mapping(source_profile_path: str, target_profile_path: str,
                             layers_target: List[int], layers_source: List[int],
                             activ_profile_lim: Optional[int] = None,
                             use_compression: bool = True) -> Dict[int, Tuple[BaseConverter, int]]:
    """
    Create layer mapping with converters using pseudoinverse method.

    Args:
        source_profile_path: Path to source model activation profile
        target_profile_path: Path to target model activation profile
        layers_target: Recipient model layers
        layers_source: Donor model layers
        activ_profile_lim: Limit on activation samples (optional)
        use_compression: Whether to use compressed storage (default: True)

    Returns:
        Dict mapping donor layers to (converter, recipient_layer) tuples
    """
    device = torch.device(get_device())

    # Get layer counts and create linear mapping
    n_source_layers = get_num_layers(source_profile_path)
    n_target_layers = get_num_layers(target_profile_path)

    # Handle layer count mismatch by truncating donor layers if needed
    if len(layers_source) > len(layers_target):
        step = len(layers_source) / len(layers_target)
        keep_indices = [min(int(i * step), len(layers_source) - 1) for i in range(len(layers_target))]
        layers_source = [layers_source[i] for i in keep_indices]
        print(f"Truncated donor layers to: {layers_source}")

    layer_mapping = create_linear_layer_mapping(layers_source, layers_target,
                                                n_source_layers, n_target_layers)
    # Create converters for each layer pair
    converter_mapping = {}
    print("Layer mapping:")

    for donor_layer, recipient_layer in sorted(layer_mapping.items()):
        # Load activations for this layer pair
        res_source, res_target = load_residual_pairs(
            source_profile_path, target_profile_path,
            donor_layer, recipient_layer, activ_profile_lim
        )

        converter = PinvConverter(
            input_dim=res_target.shape[-1],
            output_dim=res_source.shape[-1]
        )
        converter.to(device)
        converter.fit(res_target.to(device), res_source.to(device))

        # Compute loss for monitoring
        loss_value = converter.compute_loss(res_target.to(device), res_source.to(device))

        converter_mapping[donor_layer] = (converter, recipient_layer)
        print(f"  Donor layer {donor_layer} -> Recipient layer {recipient_layer} (loss: {loss_value:.8f})")

    return converter_mapping


def save_converter_mapping(converter_mapping: Dict[int, Tuple[BaseConverter, int]],
                           output_path: str, use_compression: bool = True) -> None:
    """Save converter mapping to disk with optional compression."""
    if use_compression:
        # Save in compressed format
        compressed_data = {}
        for donor_layer, (converter, recipient_layer) in converter_mapping.items():
            if hasattr(converter, 'save_compressed'):
                # For PinvConverter with compression support
                converter_data = {
                    'input_dim': converter.input_dim,
                    'output_dim': converter.output_dim,
                    'forward_matrix': converter.forward_matrix.detach().cpu(),
                    'backward_matrix': converter.backward_matrix.detach().cpu(),
                    'loss': getattr(converter, 'loss', None)
                }
                compressed_data[donor_layer] = (converter_data, recipient_layer)
            else:
                # Fallback for other converter types
                compressed_data[donor_layer] = (converter, recipient_layer)

        # Save with gzip compression
        compressed_path = output_path.replace('.pth', '_compressed.pkl.gz')
        with gzip.open(compressed_path, 'wb') as f:
            pickle.dump(compressed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Get file sizes for comparison
        original_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
        if original_size > 0:
            torch.save(converter_mapping, output_path)  # Also save uncompressed for comparison

        compressed_size = Path(compressed_path).stat().st_size

        print(f"Saved compressed converter mapping to {compressed_path}")
        if original_size > 0:
            compression_ratio = original_size / compressed_size
            print(
                f"Compression ratio: {compression_ratio:.1f}x (original: {original_size / 1e6:.1f}MB, compressed: {compressed_size / 1e6:.1f}MB)")
        else:
            print(f"Compressed size: {compressed_size / 1e6:.1f}MB")
    else:
        # Save in standard PyTorch format
        torch.save(converter_mapping, output_path)
        print(f"Saved converter mapping to {output_path}")


def load_converter_mapping(path: str) -> Dict[int, Tuple[BaseConverter, int]]:
    """Load converter mapping from disk, trying compressed format first."""
    compressed_path = path.replace('.pth', '_compressed.pkl.gz')

    device = get_device()

    if Path(compressed_path).exists():
        # Load from compressed format
        print(f"Loading compressed converters from {compressed_path}")
        with gzip.open(compressed_path, 'rb') as f:
            compressed_data = pickle.load(f)

        # Reconstruct converter mapping
        converter_mapping = {}
        for donor_layer, (converter_data, recipient_layer) in compressed_data.items():
            converter = PinvConverter(
                input_dim=converter_data['input_dim'],
                output_dim=converter_data['output_dim']
            )
            converter_mapping[donor_layer] = (converter, recipient_layer)

        return converter_mapping

    elif Path(path).exists():
        # Fall back to standard PyTorch format
        print(f"Loading converters from {path}")
        converter_mapping = torch.load(path, weights_only=False)

        return converter_mapping
    else:
        raise FileNotFoundError(f"Converter file not found: {path} or {compressed_path}")


def derive_converters(source_model: str, target_model: str,
                      source_layers: Optional[List[int]] = None,
                      target_layers: Optional[List[int]] = None,
                      save_path: Optional[str] = None,
                      activ_profile_lim: Optional[int] = None) -> dict[int, tuple[BaseConverter, int]]:
    """
    Derive converters between two models using their activation profiles.

    Args:
        source_model: Source model name (donor model)
        target_model: Target model name (recipient model)
        source_layers: Source model layers to use (default: all even layers)
        target_layers: Target model layers to use (default: all available layers)
        save_path: Path to save converters (optional)
        activ_profile_lim: Limit on activation samples to use (optional)

    Returns:
        Path where converters were saved
    """
    print(f"Deriving converters from {source_model} -> {target_model}")

    # Get clean model names
    source_name = get_model_name(source_model)
    target_name = get_model_name(target_model)

    # Get activation profile paths
    source_profile_path = str(get_activation_path(source_name))
    target_profile_path = str(get_activation_path(target_name))

    # Check if activation profiles exist
    if not Path(source_profile_path).exists():
        raise FileNotFoundError(f"Source activation profile not found: {source_profile_path}")
    if not Path(target_profile_path).exists():
        raise FileNotFoundError(f"Target activation profile not found: {target_profile_path}")

    # Get layer counts and set defaults
    n_source_layers = get_num_layers(source_profile_path)
    n_target_layers = get_num_layers(target_profile_path)

    if source_layers is None:
        source_layers = list(range(0, n_source_layers, 2))

    if target_layers is None:
        target_layers = list(range(n_target_layers))

    # Ensure we don't exceed layer counts
    source_layers = [l for l in source_layers if l < n_source_layers]
    target_layers = [l for l in target_layers if l < n_target_layers]

    print(f"Source layers ({len(source_layers)}): {source_layers}")
    print(f"Target layers ({len(target_layers)}): {target_layers}")

    # Derive converter mapping
    converter_mapping = create_converter_mapping(
        source_profile_path=source_profile_path,
        target_profile_path=target_profile_path,
        layers_target=target_layers,  # recipient layers
        layers_source=source_layers,  # donor layers
        activ_profile_lim=activ_profile_lim
    )

    if save_path is not None:
        save_converter_mapping(converter_mapping, save_path)

    return converter_mapping
