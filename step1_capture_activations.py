#!/usr/bin/env python3
"""
Command-V Step 1: Activation Profile Capture

Clean script for capturing layer activations from language models.
Supports both custom datasets and the default LIMA dataset.

Usage:
    python step1_capture_activations.py --models meta-llama/Llama-3.2-1B-Instruct
    python step1_capture_activations.py --models meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-3B-Instruct
    python step1_capture_activations.py --models meta-llama/Llama-3.2-1B-Instruct --custom-dataset prompts/test_small.json
"""

import logging
from typing import List, Optional

import fire

from commandv import capture_activations, setup_project


def main(models: List[str] = None, batch_size: int = 4,
         custom_dataset: Optional[str] = None, project_root: Optional[str] = None) -> None:
    """
    Capture activation profiles for specified models.
    
    Args:
        models: List of model identifiers to profile
        batch_size: Batch size for processing
        custom_dataset: Optional path to custom dataset (otherwise uses LIMA)
        project_root: Optional project root path
    """
    logging.basicConfig(level=logging.INFO)
    
    # Setup project paths
    if project_root:
        setup_project(project_root)
    
    # Load dataset
    if custom_dataset:
        from commandv.data import load_prompts
        prompts = load_prompts(custom_dataset)
    else:
        from commandv.data import load_lima_dataset
        prompts = load_lima_dataset()

    # Default models if none specified
    if not models:
        models = ["meta-llama/Llama-3.2-1B-Instruct"]

    # Process each model
    for model_name in models:
        from commandv.utils.paths import get_model_name, get_activation_path
        clean_name = get_model_name(model_name)
        output_file = str(get_activation_path(clean_name))
        
        capture_activations(model_name, prompts, output_file, batch_size)
        print(f"Completed activation capture for {model_name}")


if __name__ == "__main__":
    fire.Fire(main)