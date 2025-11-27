from __future__ import annotations
import math
from typing import Any, Dict
import numpy as np

def _norm_key(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return str(int(v))
        return str(v)
    return str(v)

def get_value(row, feature: str) -> Any | None:
    if hasattr(row, feature):
        val = getattr(row, feature)
    elif isinstance(row, dict) and feature in row:
        val = row[feature]
    else:
        return None
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return val

def get_modifier(modifiers: Any, value: Any) -> float:
    if not modifiers:
        return 1.0
    if isinstance(modifiers, list):
        value = int(value) if value is not None and not (isinstance(value, float) and math.isnan(value)) else None
        if value is not None:
            for age_range in modifiers:
                if age_range["min"] <= value <= age_range["max"]:
                    return age_range["mult"]
        return 1.0
    key = _norm_key(value)
    if key is not None and key in modifiers:
        return float(modifiers[key])
    return 1.0

def check_condition_match(row, feature: str, expected_values: Any) -> bool:
    v = get_value(row, feature)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    val_key = _norm_key(v)

    if isinstance(expected_values, dict) and "min" in expected_values and "max" in expected_values:
        return expected_values["min"] <= v <= expected_values["max"]

    if isinstance(expected_values, list) and len(expected_values) > 0:
        for token in expected_values:
            if isinstance(token, str) and any(op in token for op in [">=", "<=", "==", "!=", ">", "<"]):
                s = token.strip()
                # Check multi-char operators first to avoid substring conflicts
                if s.startswith(">="):
                    try:
                        return v >= float(s[2:])
                    except ValueError:
                        continue
                if s.startswith("<="):
                    try:
                        return v <= float(s[2:])
                    except ValueError:
                        continue
                if s.startswith("=="):
                    try:
                        return v == float(s[2:])
                    except ValueError:
                        continue
                if s.startswith("!="):
                    try:
                        return v != float(s[2:])
                    except ValueError:
                        continue
                if s.startswith(">"):
                    try:
                        return v > float(s[1:])
                    except ValueError:
                        continue
                if s.startswith("<"):
                    try:
                        return v < float(s[1:])
                    except ValueError:
                        continue
            else:
                if val_key == _norm_key(token):
                    return True
        return False

    return val_key == _norm_key(expected_values)