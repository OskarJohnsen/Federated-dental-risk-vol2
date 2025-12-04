from typing import List, Dict
import torch

def federated_averaging(client_weights: List[Dict], client_sample_counts: List[int]) -> Dict:
    if len(client_weights) == 0:
        raise ValueError("No client weights")
    
    if len(client_weights) != len(client_sample_counts):
        raise ValueError(f"STOP: {len(client_weights)} weights but {len(client_sample_counts)} sample counts")
    
    total_client_samples = sum(client_sample_counts)
    if total_client_samples == 0:
        raise ValueError("Client sample count = 0")
    
    aggregated_state = {}
    
    float_keys = []
    non_float_keys = []
    
    for key in client_weights[0].keys():
        tensor = client_weights[0][key]
        if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            float_keys.append(key)
        else:
            non_float_keys.append(key)
    
    for key in float_keys:
        tensor = client_weights[0][key]
        aggregated_state[key] = torch.zeros_like(tensor, dtype=torch.float32)
    
    for key in non_float_keys:
        aggregated_state[key] = client_weights[0][key].clone()

    for weights, n_samples in zip(client_weights, client_sample_counts):
        weight = n_samples / total_client_samples
        for key in float_keys:
            aggregated_state[key] += weight * weights[key].to(torch.float32)
    
    for key in float_keys:
        original_dtype = client_weights[0][key].dtype
        if original_dtype != torch.float32:
            aggregated_state[key] = aggregated_state[key].to(original_dtype)
    
    return aggregated_state

def get_model_state(model: torch.nn.Module) -> Dict:
    return model.state_dict()

def set_model_state(model: torch.nn.Module, state_dict: Dict):
    model.load_state_dict(state_dict)