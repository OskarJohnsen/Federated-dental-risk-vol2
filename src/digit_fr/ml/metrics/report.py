from typing import Dict, Any
import numpy as np
import wandb
from sklearn.metrics import f1_score
from ..constants import RISK_NAMES

def log_metrics_wandb(metrics: Dict[str, Any], prefix: str = "test/"):
    metrics_to_log = {}
    for metric_name, metric_value in metrics.items():
        if metric_name.startswith('_'):
            continue
        if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
            metrics_to_log[f"{prefix}{metric_name}"] = float(metric_value)
    
    wandb.log(metrics_to_log)

def print_dataset_metrics(dataset_metrics_dict: Dict[str, Any], risk_names: list = None):
    if risk_names is None:
        risk_names = RISK_NAMES
    
    print("\nDataset Metrics:")
    for risk_name in risk_names:
        risk_col = f"Risk_{risk_name}"
        train_count_key = f"dataset/train_{risk_col}_positive_count"
        train_ratio_key = f"dataset/train_{risk_col}_positive_ratio"
        if train_count_key in dataset_metrics_dict:
            pos_count = dataset_metrics_dict[train_count_key]
            pos_ratio = dataset_metrics_dict[train_ratio_key]
            print(f"Train - {risk_col}: {pos_count} positives ({pos_ratio:.4f})")
    
    for risk_name in risk_names:
        risk_col = f"Risk_{risk_name}"
        test_count_key = f"dataset/test_{risk_col}_positive_count"
        test_ratio_key = f"dataset/test_{risk_col}_positive_ratio"
        if test_count_key in dataset_metrics_dict:
            pos_count = dataset_metrics_dict[test_count_key]
            pos_ratio = dataset_metrics_dict[test_ratio_key]
            print(f"Test - {risk_col}: {pos_count} positives ({pos_ratio:.4f})")

def print_thresholds(optimized_thresholds: Dict[str, float], val_probs: np.ndarray, val_labels: np.ndarray, risk_names: list = None, method: str = "f1"):
    if risk_names is None:
        risk_names = RISK_NAMES
    
    for idx, risk_name in enumerate(risk_names):
        if risk_name in optimized_thresholds:
            best_thr = optimized_thresholds[risk_name]
            y_true = val_labels[:, idx]
            y_score = val_probs[:, idx]
            preds = (y_score >= best_thr).astype(int)
            
            if method.lower() == "youden":
                tp = ((preds == 1) & (y_true == 1)).sum()
                tn = ((preds == 0) & (y_true == 0)).sum()
                fp = ((preds == 1) & (y_true == 0)).sum()
                fn = ((preds == 0) & (y_true == 1)).sum()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                youden_value = sensitivity + specificity - 1
                print(f"{risk_name}: threshold={best_thr:.4f}, youden_j={youden_value:.4f}")
            else:
                best_f1 = f1_score(y_true, preds, zero_division=0)
                print(f"{risk_name}: threshold={best_thr:.4f}, f1={best_f1:.4f}")

def log_thresholds_wandb(optimized_thresholds: Dict[str, float]):
    threshold_metrics = {}
    for risk_name, threshold in optimized_thresholds.items():
        threshold_metrics[f"threshold/{risk_name}"] = threshold
    wandb.log(threshold_metrics)

def metrics_summary(final_metrics: Dict[str, Any]):
    print("METRICS SUMMARY")
    print("\nBinary Classification Metrics:")
    for metric_name, metric_value in final_metrics.items():
        if not isinstance(metric_value, (int, float)):
            continue
        if np.isnan(metric_value):
            continue
        if metric_name.startswith("roc_auc_risk_") or metric_name.startswith("pr_auc_risk_"):
            continue
        if metric_name.startswith("roc_auc_clf_") or metric_name.startswith("pr_auc_macro"):
            continue
        if metric_name.startswith("ece_risk_") or metric_name == "ece_macro":
            continue
        if metric_name.startswith("brier_score_risk_") or metric_name == "brier_score_macro":
            continue
        print(f"{metric_name}: {metric_value:.4f}")
    
    print("CALIBRATION METRICS")
    if "brier_score_macro" in final_metrics:
        print(f"Macro Average: {final_metrics['brier_score_macro']:.4f}")

    print("ECE:")
    if "ece_macro" in final_metrics:
        print(f"Macro Average: {final_metrics['ece_macro']:.4f}")

    print("ROC-AUC:")
    if "roc_auc_clf_macro" in final_metrics:
        print(f"Macro Average: {final_metrics['roc_auc_clf_macro']:.4f}")

    print("PR-AUC:")
    if "pr_auc_macro" in final_metrics:
        print(f"Macro Average: {final_metrics['pr_auc_macro']:.4f}")