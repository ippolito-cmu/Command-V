import json
import os
from typing import List, Union

import torch
from datasets import load_dataset


def tokenize_templated_prompts(prompts: List[str], tokenizer, max_length: Union[int, None],
                               padding: bool = True) -> List[torch.Tensor]:
    """
    Preprocess prompts for model input with proper chat templates.
    
    Args:
    prompts (List[str]): List of prompts
    tokenizer: Tokenizer for the model
    max_length (int): Maximum sequence length
    
    Returns:
    List[Dict]: List of preprocessed inputs
    """
    # Apply chat template for instruct models
    if hasattr(tokenizer, 'apply_chat_template') and hasattr(tokenizer,
                                                             'chat_template') and tokenizer.chat_template is not None:
        templated_prompts = apply_chat_template(prompts, tokenizer)
        return tokenizer(templated_prompts, padding=padding, truncation=True, max_length=max_length,
                         return_tensors="pt", add_special_tokens=True)
    else:
        # Fallback for models without chat templates
        return tokenizer(prompts, padding=padding, truncation=True, max_length=max_length,
                         return_tensors="pt", add_special_tokens=True)


def apply_chat_template(prompts: List[str], tokenizer) -> List[str]:
    """
    Apply chat template to prompts for instruct models.
    
    Args:
    prompts (List[str]): List of prompts
    tokenizer: Tokenizer for the model
    
    Returns:
    List[str]: Prompts with chat template applied
    """
    return [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        for prompt in prompts]


def load_lima_dataset(hf_token=None):
    dataset = load_dataset('GAIR/lima', hf_token=hf_token)['train']
    return [conversation[0] for conversation in dataset['conversations']]


def load_prompts(input_source: Union[List[str], str]) -> List[str]:
    """
    Load prompts from various input sources.

    Args:
        input_source: Either a list of strings, path to JSON file, or path to a text file with one prompt per line

    Returns:
        List of prompts
    """

    if isinstance(input_source, list):
        return input_source

    # Check if input is a file path
    if not isinstance(input_source, str) or not os.path.exists(input_source):
        raise ValueError(f"Input source {input_source} not found or not supported")

    # Handle JSON file
    if input_source.endswith(".json"):
        with open(input_source, "r") as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return data
            if all(isinstance(item, list) for item in data) and all(isinstance(item, str) for item in data[0]):
                return [l[0] for l in data]
            elif all(isinstance(item, dict) for item in data):
                # Try to extract prompts from common field names
                for field in ["prompt", "instruction", "input", "text", "query"]:
                    if all(field in item for item in data):
                        print(f"Loaded `{field}` from the file")
                        return [item[field] for item in data]

                raise ValueError(f"Cannot determine prompt field in JSON dictionary items")
        elif isinstance(data, dict) and "prompts" in data:
            return data["prompts"]

        raise ValueError(f"Unsupported JSON format in {input_source}")

    # Handle texte file (one prompt per line)
    with open(input_source, "r") as f:
        return [line.strip() for line in f if line.strip()]
