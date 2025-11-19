from typing import Dict, Optional
import numpy as np
from sklearn.metrics import f1_score, roc_curve
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