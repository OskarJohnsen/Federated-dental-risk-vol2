from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from .common import get_modifier, get_value, check_condition_match

def apply_modifiers(modifiers_dict: Any, row: Dict[str, Any], feature_name: str) -> float:
    value = get_value(row, feature_name)
    if value is None:
        return 1.0
    return get_modifier(modifiers_dict, value)

def compute_removal_decision(row: Dict[str, Any], binary_cfg: Dict[str, Any]) -> Tuple[int, float]:
    removal_prob = float(binary_cfg["base_prevalence"])

    for feature, modifiers in binary_cfg["positive_indications"].items():
        value = get_value(row, feature)
        mult = get_modifier(modifiers, value)
        removal_prob *= mult

    for feature, modifiers in binary_cfg["counter_indications"].items():
        if feature == "Systemic_Risk":
            for risk_factor, risk_modifiers in modifiers.items():
                mult = apply_modifiers(risk_modifiers, row, risk_factor)
                removal_prob *= mult
        else:
            mult = apply_modifiers(modifiers, row, feature)
            removal_prob *= mult

    for feature, modifiers in binary_cfg.get("contextual_modifiers", {}).items():
        mult = apply_modifiers(modifiers, row, feature)
        removal_prob *= mult

    for interaction in binary_cfg.get("interactions", []):
        conditions = interaction["conditions"]
        match = all(check_condition_match(row, feature, expected_values) for feature, expected_values in conditions.items())
        if match:
            removal_prob *= interaction["mult"]

    removal_prob = min(1.0, max(0.0, removal_prob))
    removal_decision = int(np.random.rand() < removal_prob)
    return removal_decision, float(removal_prob)