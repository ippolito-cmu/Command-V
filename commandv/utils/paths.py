"""
Path management for Command-V.
Provides clean, configurable path resolution for all components.
"""

import os
from pathlib import Path
from typing import Optional, Union


class ProjectPaths:
    """Clean path management with environment variable support."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize paths with project root."""
        self.project_root = self._find_project_root(project_root)
        self._setup_directories()

    def _find_project_root(self, provided_root: Optional[Path] = None) -> Path:
        """Find project root intelligently."""
        if provided_root:
            return Path(provided_root).resolve()

        # Find by looking for config.yaml or setup.py
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if any((parent / marker).exists() for marker in ['config.yaml', 'setup.py', 'pyproject.toml']):
                return parent

        # Fallback to current working directory
        return Path.cwd()

    def _setup_directories(self):
        """Setup directory structure with environment variable support."""
        self.root = self.project_root
        self.outputs = self.root / 'outputs'
        self.prompts = self.root / 'prompts'
        self.adapters = self.root / 'reft-adapters'

        # Support external activations directory via environment variable
        activations_env = os.getenv('COMMANDV_ACTIVATIONS_PATH')
        if activations_env and Path(activations_env).exists():
            self.activations = Path(activations_env)
            print(f"Using external activations: {self.activations}")
        else:
            self.activations = self.outputs / 'activations'

        self.converters = self.outputs / 'converters'
        self.inferences = self.outputs / 'inferences'

        # Create directories as needed
        directories_to_create = [self.outputs, self.converters, self.inferences]
        if self.activations == self.outputs / 'activations':
            directories_to_create.append(self.activations)

        for directory in directories_to_create:
            directory.mkdir(parents=True, exist_ok=True)


# Global instance
_paths = None


def get_paths() -> ProjectPaths:
    """Get the global paths instance."""
    global _paths
    if _paths is None:
        _paths = ProjectPaths()
    return _paths


def setup_project(project_root: Optional[str] = None) -> ProjectPaths:
    """Setup project with custom root if needed."""
    global _paths
    _paths = ProjectPaths(Path(project_root) if project_root else None)
    return _paths


def get_model_name(model_path_or_id: Union[str, Path]) -> str:
    """Extract clean model name from path or HuggingFace ID."""
    return Path(model_path_or_id).name


def get_activation_path(model_name: str, aggregation: str = "last") -> Path:
    """Get path for activation files with fallback naming support."""
    paths = get_paths()

    # Try multiple naming conventions for compatibility
    possible_names = [
        f"Layer.out_{aggregation}_{model_name}_bf16.safetensors",
        f"activations_Layer.out_{aggregation}_{model_name}_bf16.safetensors",
    ]

    for filename in possible_names:
        file_path = paths.activations / filename
        if file_path.exists():
            return file_path

    # Return preferred format if none found
    return paths.activations / possible_names[0]


def get_converter_path(source_model: str, target_model: str, layers: str) -> Path:
    """Get path for converter files."""
    paths = get_paths()
    filename = f"{source_model}_to_{target_model}_PinvConverter_L{layers}.pth"
    return paths.converters / filename



