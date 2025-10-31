from __future__ import annotations
import math
from typing import Dict, Any
from .common import check_condition_match, get_modifier, get_value

def compute_risk_from_evidence(row: Dict[str, Any], risk_type: str, risk_config: Dict[str, Any], surgical_decision: int | None = None) -> float:
    """Apply feature modifiers"""
    risk_data = risk_config[risk_type]
    risk = float(risk_data["base_incidence"])

    feature_mults: Dict[str, float] = {}
    for feature, modifiers in risk_data.get("risk_modifiers", {}).items():
        mult = float(get_modifier(modifiers, get_value(row, feature)))
        feature_mults[feature] = mult
        risk *= mult

    for interaction in risk_data.get("interactions", []):
        conditions = interaction.get("conditions", {})
        if all(check_condition_match(row, f, v) for f, v in conditions.items()):
            raw_mult = float(interaction["mult"])
            prod_features = 1.0
            for f in conditions.keys():
                prod_features *= feature_mults.get(f, 1.0)
            adj_mult = raw_mult / prod_features if prod_features > 0 else raw_mult
            adj_mult = max(0.1, min(10.0, adj_mult))
            risk *= adj_mult

    if surgical_decision is not None and "surgery_modifiers" in risk_data:
        mult = float(get_modifier(risk_data["surgery_modifiers"], surgical_decision))
        risk *= mult

    if risk_type == "NerveDysesthesia":
        prox_nerve = get_value(row, "Proximity_Nerve")
        if prox_nerve is not None and prox_nerve == 0:
            risk *= 0.30
        mandi_maxi = get_value(row, "Mandi_Maxi")
        if mandi_maxi is not None and mandi_maxi == 1:
            risk = 0.0

    result = float(min(1.0, max(0.0, risk)))
    return result