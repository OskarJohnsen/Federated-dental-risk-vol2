from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np

def generate_client_profiles(n_profiles: int, ranges_config: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    if seed is not None:
        np.random.seed(seed)
    
    profiles: Dict[str, Dict[str, Any]] = {}
    
    age_range = ranges_config.get("age_mu", {"min": 20, "max": 40})
    proximity_range = ranges_config.get("proximity_nerve_p", {"min": 0.15, "max": 0.60})
    score_scale_range = ranges_config.get("score_scale", {"min": 0.70, "max": 1.30})
    missingness_rate_range = ranges_config.get("missingness_rate", {"min": 0.0, "max": 0.25})
    noise_range = ranges_config.get("measurement_noise", {"min": 0.03, "max": 0.15})
    max_missing_cols = ranges_config.get("max_missingness_columns", 3)
    
    missingness_eligible_cols = ["Tooth_Mobility", "Periodontal_Status", "Root_Development", "Tooth_Angulation", "Bone_Density"]
    
    for client_id_int in range(1, n_profiles + 1):
        client_id = str(client_id_int)
        age_mu = np.random.uniform(age_range["min"], age_range["max"])
        proximity_p = np.random.uniform(proximity_range["min"], proximity_range["max"])
        
        impaction_alpha = np.random.uniform(0.5, 2.0, size=3)
        impaction_probs = np.random.dirichlet(impaction_alpha)
        impaction_depth = impaction_probs.tolist()
        
        prevalence_shift = {
            "Age_mu": float(age_mu),
            "Proximity_Nerve_p": float(proximity_p),
            "Impaction_Depth": [float(x) for x in impaction_depth]
        }
        
        score_scale = {}
        for risk_id in range(1, 5):
            scale = np.random.uniform(score_scale_range["min"], score_scale_range["max"])
            score_scale[risk_id] = float(scale)
        
        n_missing_cols = np.random.randint(0, max_missing_cols + 1)
        if n_missing_cols > 0:
            selected_cols = np.random.choice(missingness_eligible_cols, size=min(n_missing_cols, len(missingness_eligible_cols)), replace=False)
            missingness = {}
            for col in selected_cols:
                rate = np.random.uniform(missingness_rate_range["min"], missingness_rate_range["max"])
                missingness[col] = float(rate)
        else:
            missingness = {}
        
        measurement_noise = np.random.uniform(noise_range["min"], noise_range["max"])
        
        profiles[client_id] = {
            "name": f"Clinic_{client_id}",
            "prevalence_shift": prevalence_shift,
            "score_scale": {str(k): v for k, v in score_scale.items()},
            "missingness": missingness,
            "measurement_noise": float(measurement_noise)
        }
    print("Profile generator used")
    return profiles