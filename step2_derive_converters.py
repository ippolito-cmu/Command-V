#!/usr/bin/env python3
"""
Command-V Step 2: Converter Derivation

Clean script for deriving converters between model activation profiles.
Only supports PinvConverter with linear layer mapping.

Usage:
    python step2_derive_converters.py derive --source-model meta-llama/Llama-3.1-8B-Instruct --target-model meta-llama/Llama-3.2-1B-Instruct
    python step2_derive_converters.py list
"""

import logging
from typing import List, Optional

import fire

from commandv import derive_converters, setup_project, get_paths


def derive(source_model: str, target_model: str,
           source_layers: Optional[List[int]] = None,
           target_layers: Optional[List[int]] = None,
           save_path: Optional[str] = None,
           activ_profile_lim: Optional[int] = None,
           project_root: Optional[str] = None) -> str:
    """
    Derive converters between two models using their activation profiles.
    
    Args:
        source_model: Source model name (donor model)
        target_model: Target model name (recipient model)
        source_layers: Source model layers to use (default: all even layers)
        target_layers: Target model layers to use (default: all available layers)
        save_path: Path to save converters (optional)
        activ_profile_lim: Limit on activation samples to use (optional)
        project_root: Optional project root path
    
    Returns:
        Path where converters were saved
    """
    # Setup project paths
    if project_root:
        setup_project(project_root)
        
    return derive_converters(
        source_model=source_model,
        target_model=target_model,
        source_layers=source_layers,
        target_layers=target_layers,
        save_path=save_path,
        activ_profile_lim=activ_profile_lim
    )


def list_profiles() -> None:
    """List available activation profiles."""
    paths = get_paths()
    
    print("Available activation profiles:")
    for profile_file in paths.activations.glob("*.safetensors"):
        # Extract model name from filename
        name_parts = profile_file.stem.split('_')
        if len(name_parts) >= 3:
            model_name = name_parts[2]  # Assuming format: Layer.out_last_ModelName_bf16
            print(f"  - {model_name}: {profile_file}")


def list_converters() -> None:
    """List available converter mappings."""
    paths = get_paths()
    
    print("Available converter mappings:")
    for converter_file in paths.converters.glob("*.pth"):
        print(f"  - {converter_file.stem}: {converter_file}")


def main():
    """Main CLI interface."""
    logging.basicConfig(level=logging.INFO)
    
    commands = {
        'derive': derive,
        'list': list_profiles,
        'list-converters': list_converters,
    }
    
    fire.Fire(commands)


if __name__ == "__main__":
    main()