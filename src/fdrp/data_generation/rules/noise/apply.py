from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any

def add_feature_noise(df: pd.DataFrame, client_id: int, client_profiles: Dict[int, Any], noise_config: Dict[str, Any]) -> pd.DataFrame:
    prof = client_profiles.get(client_id, {})
    noise_level = prof.get("measurement_noise", 0.05)

    client_mask = df["Client"] == client_id

    # Ordinal features
    ordinal_config = noise_config["ordinal_features"]
    for feat, params in ordinal_config.items():
        min_val = params["min"]
        max_val = params["max"]
        prob = noise_level * params["noise_multiplier"]
        mask = client_mask & (np.random.rand(len(df)) < prob)
        shift = np.random.choice([-1, 1], size=mask.sum())
        df.loc[mask, feat] = np.clip(df.loc[mask, feat] + shift, min_val, max_val)

    # Binary features
    binary_config = noise_config["binary_features"]
    for feature, params in binary_config.items():
        prob = noise_level * params["noise_multiplier"]
        mask = client_mask & (np.random.rand(len(df)) < prob)
        df.loc[mask, feature] = 1 - df.loc[mask, feature]

    return df