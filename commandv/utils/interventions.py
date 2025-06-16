"""
Intervention loading utilities for Command-V.
Handles loading ReFT interventions from saved adapters.
"""

import json
import pathlib
from collections import defaultdict
from typing import Dict, List

import torch
from pyvene import RepresentationConfig

from .types import ReconstructedNodireftIntervention


def load_interventions(adapter_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> Dict[int, List]:
    """
    Load interventions from ReFT adapter folder.
    
    Args:
        adapter_path: Path to adapter folder
        device: Target device for interventions
        dtype: Target dtype for interventions (default: bfloat16)
        
    Returns:
        Dict mapping layer indices to lists of interventions
    """
    config = json.load(open(pathlib.Path(adapter_path) / "config.json"))
    interventions = defaultdict(list)

    for name, _, rep, _ in zip(
            config['sorted_keys'], config['intervention_types'],
            config['representations'], config['intervention_dimensions']):
        cfg = RepresentationConfig(*rep)
        state_dict = dict(torch.load(pathlib.Path(adapter_path) / f"intkey_{name}.bin", weights_only=True))

        embed_dim = state_dict.pop('embed_dim').item()
        interchange_dim = state_dict.pop('interchange_dim').item()

        intervention = ReconstructedNodireftIntervention(
            embed_dim=embed_dim, interchange_dim=interchange_dim,
            low_rank_dimension=cfg.low_rank_dimension, state_dict=state_dict
        ).eval().to(device=device, dtype=dtype)

        interventions[cfg.layer].append(intervention)

    return interventions