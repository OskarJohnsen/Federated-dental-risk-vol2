from typing import Dict, Optional
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.metrics import f1_score, roc_curve
from ...core.paths import root_path
from ..constants import RISK_NAMES

def optimize_thresholds_f1(val_probs: np.ndarray, val_labels: np.ndarray, risk_names: list = None) -> Dict[str, float]:
    if risk_names is None:
        risk_names = RISK_NAMES
    
    optimized_thresholds = {}
    if val_probs is not None and val_labels is not None and len(risk_names) > 0:
        thresholds = np.linspace(0.01, 0.99, 99)
        
        for idx, risk_name in enumerate(risk_names):
            y_true = val_labels[:, idx]
            y_score = val_probs[:, idx]
            
            best_thr = 0.5
            best_f1 = 0.0
            
            for thr in thresholds:
                preds = (y_score >= thr).astype(int)
                if preds.sum() == 0:
                    continue
                
                f1 = f1_score(y_true, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = float(thr)
            
            optimized_thresholds[risk_name] = best_thr
    
    return optimized_thresholds

def optimize_thresholds_youden(val_probs: np.ndarray, val_labels: np.ndarray, risk_names: list = None) -> Dict[str, float]:
    if risk_names is None:
        risk_names = RISK_NAMES
    
    optimized_thresholds = {}
    if val_probs is not None and val_labels is not None and len(risk_names) > 0:
        for idx, risk_name in enumerate(risk_names):
            y_true = val_labels[:, idx]
            y_score = val_probs[:, idx]
            
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            
            youden_j = tpr - fpr
            
            optimal_idx = np.argmax(youden_j)
            best_thr = float(thresholds[optimal_idx])
            
            if not np.isfinite(best_thr):
                best_thr = 0.5
            
            optimized_thresholds[risk_name] = best_thr
    
    return optimized_thresholds

def percentile_thresholds(val_probs: np.ndarray, percentiles: list = [33, 67], risk_names: list = None) -> Dict[str, Dict[str, float]]:
    if risk_names is None:
        risk_names = RISK_NAMES
    
    thresholds = {}
    
    sorted_percentiles = sorted(percentiles)
    
    for idx, risk_name in enumerate(risk_names):
        val_prob_values = val_probs[:, idx]
        
        if len(val_prob_values) > 0:
            percentile_values = np.percentile(val_prob_values, sorted_percentiles)
            
            risk_thresholds = {}
            if len(sorted_percentiles) == 2:
                risk_thresholds["low_medium_boundary"] = float(percentile_values[0])
                risk_thresholds["medium_high_boundary"] = float(percentile_values[1])
            else:
                for i, pct in enumerate(sorted_percentiles):
                    risk_thresholds[f"percentile_{pct}"] = float(percentile_values[i])
            
            thresholds[risk_name] = risk_thresholds
        else:
            print(f"No val_probs found: {len(val_prob_values)}")
            thresholds[risk_name] = {
                "low_medium_boundary": 0.33,
                "medium_high_boundary": 0.67
            }
    
    return thresholds

def apply_risk_categorization(probs: np.ndarray, thresholds: Dict[str, Dict[str, float]], risk_names: list = None ) -> np.ndarray:
    if risk_names is None:
        risk_names = RISK_NAMES
    
    categories = np.zeros_like(probs, dtype=int)
    
    for idx, risk_name in enumerate(risk_names):
        risk_probs = probs[:, idx]
        risk_thresholds = thresholds[risk_name]
        
        if "low_medium_boundary" in risk_thresholds and "medium_high_boundary" in risk_thresholds:
            low_med = risk_thresholds["low_medium_boundary"]
            med_high = risk_thresholds["medium_high_boundary"]

            categories[:, idx] = np.where(risk_probs < low_med,  0, np.where(risk_probs < med_high, 1, 2))
        else:
            low_med = 0.33
            med_high = 0.67
            categories[:, idx] = np.where(risk_probs < low_med, 0, np.where(risk_probs < med_high, 1, 2))
    
    return categories

def load_global_thresholds(thresholds_path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    if thresholds_path is None:
        thresholds_path = root_path("configs", "global_thresholds.json")
    else:
        thresholds_path = Path(thresholds_path)
    
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Not found: {thresholds_path}")
    
    with thresholds_path.open("r") as f:
        raw_thresholds = json.load(f)
    
    converted_thresholds = {}
    for risk_name, thresholds in raw_thresholds.items():
        converted_thresholds[risk_name] = {
            "low_medium_boundary": thresholds.get("low_medium", 0.33),
            "medium_high_boundary": thresholds.get("medium_high", 0.67)
        }
    
    return converted_thresholds