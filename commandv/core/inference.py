"""
Command-V inference functionality.
Provides clean interface for behavioral transfer during inference.
"""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.processing import tokenize_templated_prompts
from ..utils.interventions import load_interventions
from ..utils.paths import get_model_name, get_paths
from ..utils.types import BaseConverter

# Configuration constants
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_TOKENS = 256


class CommandVInference:
    """Clean Command-V inference engine for behavioral transfer."""

    def __init__(self, recipient_model_id: str, donor_model_id: str,
                 adapter_folder: str, converters: Optional[str] = None, model=None):
        """
        Initialize Command-V inference.
        
        Args:
            recipient_model_id: HuggingFace model ID for recipient
            donor_model_id: HuggingFace model ID for donor  
            adapter_folder: Path to ReFT adapter folder
            converters: converters maps (optional, will auto-detect)
        """
        self.recipient_model_id = recipient_model_id
        self.donor_model_id = donor_model_id
        self.adapter_folder = adapter_folder

        # Load models
        print(f"Loading recipient model: {recipient_model_id}")
        self.model = model or  AutoModelForCausalLM.from_pretrained(
            recipient_model_id,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(recipient_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load interventions
        print(f"Loading interventions from: {adapter_folder}")
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.interventions = load_interventions(adapter_folder, device, dtype)
        self.layer_mappings = converters

        print("Command-V setup complete. Ready for inference.")

    def _find_converter_path(self) -> str:
        """Auto-detect converter path based on model names."""
        donor_name = get_model_name(self.donor_model_id)
        recipient_name = get_model_name(self.recipient_model_id)

        # Try to find existing converter file
        paths = get_paths()

        for converter_file in paths.converters.glob("*.pth"):
            if donor_name in converter_file.name and recipient_name in converter_file.name:
                return str(converter_file)

        raise FileNotFoundError(
            f"No converter found for {donor_name} -> {recipient_name}. "
            f"Please run: commandv.derive_converters('{self.donor_model_id}', '{self.recipient_model_id}')"
        )

    def _create_hook_function(self, converter: BaseConverter, intervention):
        """Create hook function for Command-V intervention."""

        def hook_fn(module, input, output):
            hidden_state = output[0]
            if hidden_state.shape[1] == 1:
                return output  # Skip during generation

            modified_hidden_state = hidden_state.clone()
            batch_size = hidden_state.shape[0]

            for i in range(batch_size):
                last_token_pos = hidden_state.shape[1] - 1
                last_token_state = hidden_state[i, last_token_pos].unsqueeze(0)

                # Core Command-V operation: convert -> intervene -> convert back
                # Use target_dtype if available, otherwise use forward_matrix dtype
                target_dtype = getattr(converter, 'target_dtype', converter.forward_matrix.dtype)
                last_token_state = last_token_state.to(converter.forward_matrix.device, dtype=target_dtype)
                converted_state = converter.forward_transform(last_token_state)
                intervened_state = intervention(converted_state)
                bias = converter.backward_transform(intervened_state)

                bias = bias.to(hidden_state.device)
                modified_hidden_state[i, last_token_pos] += bias.squeeze(0)

            return (modified_hidden_state,) + output[1:]

        return hook_fn

    def generate(self, prompts: List[str], max_new_tokens: int = DEFAULT_MAX_TOKENS,
                 batch_size: int = DEFAULT_BATCH_SIZE) -> List[str]:
        """
        Generate text with Command-V behavioral transfer.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for processing
            
        Returns:
            List of generated texts
        """
        # Setup hooks for behavioral transfer
        hooks = []
        handles = []

        for donor_layer, layer_interventions in self.interventions.items():
            if donor_layer in self.layer_mappings:
                converter, recipient_layer = self.layer_mappings[donor_layer]
                for intervention in layer_interventions:
                    hook_fn = self._create_hook_function(converter, intervention)
                    target_layer = (
                        self.model.module.model.layers[recipient_layer]
                        if hasattr(self.model, "module")
                        else self.model.model.layers[recipient_layer]
                    )
                    handle = target_layer.register_forward_hook(hook_fn)
                    handles.append(handle)

        try:
            results = []

            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]

                # Tokenize
                inputs = tokenize_templated_prompts(batch, self.tokenizer, max_length=1024)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                # Decode
                for j, output in enumerate(outputs):
                    input_len = inputs['input_ids'][j].shape[0]
                    generated_tokens = output[input_len:]
                    text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    results.append(text)

            return results

        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()
