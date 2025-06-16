"""
Command-V activation capture functionality.
Provides clean interface for profiling model activations.
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.processing import tokenize_templated_prompts, apply_chat_template, load_lima_dataset
from ..utils.paths import get_activation_path, get_model_name
from ..utils.hooks import hook_model_layers, remove_hooks, get_last_token_activations
from ..utils.activations import get_layer_name

# Configuration constants
TOKENIZER_MAX_LEN = 1024
DEFAULT_BATCH_SIZE = 4


def capture_activations(model_name: str, prompts: List[str], output_file: str, 
                       batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """
    Capture and save activation profiles for a model.
    
    Args:
        model_name: HuggingFace model identifier
        prompts: List of text prompts for profiling
        output_file: Path to save activations (.safetensors)
        batch_size: Processing batch size
    """
    if Path(output_file).exists():
        logging.info(f"Output file {output_file} already exists. Skipping.")
        return

    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply chat template for instruct models
    if "instruct" in model_name.lower() or "it" in model_name.lower():
        prompts = apply_chat_template(prompts, tokenizer)
    
    print(f"Processing {len(prompts)} prompts with batch size {batch_size}")

    # Capture activations layer by layer
    layer_names = [get_layer_name(i) for i in range(model.config.num_hidden_layers)]
    all_activations = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        torch.cuda.empty_cache()
        
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenize_templated_prompts(batch_prompts, tokenizer, max_length=TOKENIZER_MAX_LEN).to(
            next(model.parameters()).device)
        
        # Get last token positions
        seq_length = inputs['attention_mask'].size(1)
        position_indices = torch.arange(seq_length, device=inputs['attention_mask'].device)
        last_indices = (inputs['attention_mask'] * position_indices).max(1).values

        # Hook layers and run forward pass
        hooks, handles = hook_model_layers(model, layer_names, "Layer.out", dtype=torch.bfloat16)
        
        with torch.no_grad():
            _ = model(**inputs)
        
        activations = {name: hook.activation for name, hook in hooks.items()}
        remove_hooks(handles)

        # Extract last token activations
        token_activations = get_last_token_activations(activations, last_indices=last_indices)
        all_activations.append({k: v.cpu() for k, v in token_activations.items()})

    # Combine all batches
    combined_activations = {
        layer: torch.cat([batch[layer] for batch in all_activations]) 
        for layer in layer_names
    }

    # Save with metadata
    metadata = {
        'model_name': str(model_name),
        'num_layers': str(model.config.num_hidden_layers),
        'hidden_size': str(model.config.hidden_size),
        'num_prompts': str(len(prompts)),
        'activation_point': 'Layer.out',
        'aggregation': 'last',
    }

    save_file(combined_activations, output_file, metadata=metadata)
    print(f"Activations saved to {output_file}")


def profile_models(models: List[str] = None, batch_size: int = DEFAULT_BATCH_SIZE,
                   custom_dataset: Optional[str] = None) -> None:
    """
    Run activation profiling on multiple models.
    
    Args:
        models: List of model identifiers to profile
        batch_size: Batch size for processing
        custom_dataset: Optional path to custom dataset (otherwise uses LIMA)
    """
    logging.basicConfig(level=logging.INFO)

    # Load dataset
    if custom_dataset:
        from ..data.processing import load_prompts
        prompts = load_prompts(custom_dataset)
    else:
        prompts = load_lima_dataset()

    # Default models if none specified
    if not models:
        models = ["meta-llama/Llama-3.2-1B-Instruct"]

    for model_name in models:
        clean_name = get_model_name(model_name)
        output_file = str(get_activation_path(clean_name))
        
        capture_activations(model_name, prompts, output_file, batch_size)
        print(f"Completed activation capture for {model_name}")