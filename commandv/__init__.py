"""
Command-V: Pasting LLM Behaviors via Activation Profiles

A simplified implementation of behavioral transfer using activation profiles
without backpropagation. Core functionality includes:

- Activation profiling
- Converter derivation
- Behavioral inference

Usage:
    from commandv import capture_activations, derive_converters, CommandVInference
"""

from .core.capture import capture_activations
from .core.converters import derive_converters, PinvConverter
from .core.inference import CommandVInference
from .utils.paths import setup_project, get_paths

__version__ = "1.0.0"
__all__ = [
    "capture_activations",
    "derive_converters", 
    "CommandVInference",
    "PinvConverter",
    "setup_project",
    "get_paths"
]