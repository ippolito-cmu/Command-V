#!/usr/bin/env python3
"""
Command-V Step 3: Behavioral Transfer Inference

Clean script for performing inference with behavioral transfer using cached converters.

Usage:
    python step3_commandv_inference.py --recipient-model meta-llama/Llama-3.2-1B-Instruct --donor-model meta-llama/Llama-3.1-8B-Instruct --adapter-folder reft-adapters/jailbreak/Llama-3.1-8B-Instruct/NodireftIntervention/l1/walledai--AdvBench/L0;2;4;6;8;10;12;14;16;18;20;22;24;26;28;30 --input-source prompts/AdvBench/test.txt --first-n 5
"""

import logging
from typing import Optional

import fire

from commandv import CommandVInference, setup_project
from commandv.data import load_prompts


def main(recipient_model: str, donor_model: str, adapter_folder: str,
         input_source: str, first_n: int = -1, max_new_tokens: int = 256,
         batch_size: int = 1, converter_path: Optional[str] = None,
         print_output_only: bool = False, project_root: Optional[str] = None):
    """
    Run Command-V inference with behavioral transfer.
    
    Args:
        recipient_model: Recipient model identifier
        donor_model: Donor model identifier
        adapter_folder: Path to ReFT adapter folder
        input_source: Path to input prompts file
        first_n: Limit to first N prompts (-1 for all)
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
        converter_path: Path to saved converters (optional)
        print_output_only: Only print outputs, don't save
        project_root: Optional project root path
    """
    logging.basicConfig(level=logging.INFO)
    
    # Setup project paths
    if project_root:
        setup_project(project_root)
    
    # Load prompts
    prompts = load_prompts(input_source)
    if first_n > 0:
        prompts = prompts[:first_n]
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Initialize Command-V
    cmd_v = CommandVInference(
        recipient_model_id=recipient_model,
        donor_model_id=donor_model,
        adapter_folder=adapter_folder,
        converters=converter_path
    )
    
    # Generate with proper templates and tokens
    print(f"Generating responses with max_new_tokens={max_new_tokens}")
    results = cmd_v.generate(prompts, max_new_tokens, batch_size)
    
    # Output results
    if print_output_only:
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\\n=== Prompt {i+1} ===")
            print(f"Input: {prompt}")
            print(f"Output: {result}")
    else:
        # Save to CSV
        import csv
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"commandv_results_{timestamp}.csv"
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt', 'output'])
            for prompt, result in zip(prompts, results):
                writer.writerow([prompt, result])
        
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    fire.Fire(main)