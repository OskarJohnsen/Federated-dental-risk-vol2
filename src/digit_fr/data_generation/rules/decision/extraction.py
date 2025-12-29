from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from .common import check_condition_match, get_value

def get_base_priors(extraction_cfg: Dict[str, Any]) -> Dict[int, float]:
    return {int(k): float(v) for k, v in extraction_cfg["_base_priors"].items()}

def add(scores: Dict[int, float], cls: int, amt: float) -> None:
    scores[cls] = scores.get(cls, 0.0) + float(amt)

def evaluate_rule_conditions(row: Dict[str, Any], rule_data: Dict[str, Any]) -> bool:
    if "conditions" not in rule_data:
        return False
    conditions = rule_data["conditions"]
    if not all(check_condition_match(row, feature, expected_values) for feature, expected_values in conditions.items()):
        return False
    if "additional_conditions" in rule_data:
        additional = rule_data["additional_conditions"]
        if not all(check_condition_match(row, feature, values) for feature, values in additional.items()):
            return False
    return True

def apply_rule_category(row: Dict[str, Any], scores: Dict[int, float], rule_category: str, config: Dict[str, Any]) -> Dict[int, float]:
    if rule_category not in config:
        return scores
    for rule_name, rule_data in config[rule_category].items():
        if isinstance(rule_data, dict) and not any(key in rule_data for key in ["conditions", "effects", "description"]):
            value = get_value(row, rule_name)
            if value is not None and str(value) in rule_data:
                for cls, effect in rule_data[str(value)]["effects"].items():
                    add(scores, int(cls), effect)
        elif rule_category == "age_rules" and rule_name == "Age_Scaling":
            age = get_value(row, "Age")
            if age is not None:
                age_factor = max(0.0, min(2.0, (age - 25) / 10.0))
                for cls, effect in rule_data.get("effects", {}).items():
                    add(scores, int(cls), effect * age_factor)
        elif evaluate_rule_conditions(row, rule_data):
            for cls, effect in rule_data["effects"].items():
                add(scores, int(cls), effect)
    return scores

# removed class 4: odontectomy
def compute_scores_row(row: Dict[str, Any], extraction_cfg: Dict[str, Any], client_profiles: Dict[int, Any], iid_type: str = "non-iid") -> Dict[int, float]:
    base_priors = get_base_priors(extraction_cfg)
    s = {1: base_priors[1], 2: base_priors[2], 3: base_priors[3]}

    s = apply_rule_category(row, s, "symptom_rules", extraction_cfg)
    s = apply_rule_category(row, s, "anatomy_rules", extraction_cfg)
    s = apply_rule_category(row, s, "pathology_rules", extraction_cfg)
    s = apply_rule_category(row, s, "mobility_rules", extraction_cfg)
    s = apply_rule_category(row, s, "root_development_rules", extraction_cfg)
    s = apply_rule_category(row, s, "periodontal_rules", extraction_cfg)
    s = apply_rule_category(row, s, "caries_rules", extraction_cfg)
    s = apply_rule_category(row, s, "systemic_rules", extraction_cfg)
    s = apply_rule_category(row, s, "age_rules", extraction_cfg)
    s = apply_rule_category(row, s, "ian_proximity_rules", extraction_cfg)
    s = apply_rule_category(row, s, "symptom_interactions", extraction_cfg)

    return s

def decide_row(row: Dict[str, Any], extraction_cfg: Dict[str, Any], client_profiles: Dict[int, Any], temperature: float, noise_sd: float, iid_type: str = "non-iid") -> Tuple[int, Dict[int, float], np.ndarray]:
    s = compute_scores_row(row, extraction_cfg, client_profiles, iid_type)
    for k in s:
        s[k] += np.random.normal(0.0, noise_sd)
    logits = np.array([s[1], s[2], s[3]], dtype=float)
    logits = logits - logits.max()
    ex = np.exp(logits / max(1e-6, temperature))
    probs = ex / ex.sum()
    decision = int(np.random.choice([1, 2, 3], p=probs))
    return decision, s, probs