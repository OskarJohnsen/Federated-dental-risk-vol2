from typing import Dict, Any
import torch
import numpy as np
import pandas as pd
from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc
from sklearn.metrics import matthews_corrcoef, brier_score_loss, average_precision_score, confusion_matrix
from sklearn.metrics import f1_score as sk_f1_score
from ..constants import RISK_NAMES

# How often do clinics disagree? -> percent
def inconsistency_any(categories: np.ndarray) -> float:
    n, k = categories.shape
    pairs = [(a, b) for a in range(k) for b in range(a+1, k)]
    total = n * len(pairs)
    disagreement = 0
    for a, b in pairs:
        disagreement += np.sum(categories[:, a] != categories[:, b])
    return disagreement / total

# When they disagree, how severe is the disagreement? -> percent of max_disagreement (if cats = low, medium, high then max_disagreement = high - low = 2)
def inconsistency_distance(categories: np.ndarray) -> float:
    n, k = categories.shape
    pairs = [(a, b) for a in range(k) for b in range(a+1, k)]
    max_disagreement = categories.max() - categories.min()
    if max_disagreement == 0:
        return 0.0

    total = n * len(pairs) * max_disagreement
    dist_sum = 0
    for a, b in pairs:
        dist_sum += np.abs(categories[:, a] - categories[:, b]).sum()
    return dist_sum / total

# What proportion of patients would potentially receive at least one different risk category depending on which clinic they go to?
def patient_disagreement(categories: np.ndarray) -> float:
    n, _ = categories.shape
    all_same = np.all(categories == categories[:, [0]], axis=1)

    num_disagree_patients = np.sum(~all_same)
    n = categories.shape[0]

    return num_disagree_patients / n

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_bin = in_bin.mean()
        
        if prop_bin > 0:
            accuracy_bin = labels[in_bin].mean()
            avg_confidence_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_bin - accuracy_bin) * prop_bin
    
    return float(ece)

def ece_probability(pred_probs: np.ndarray, true_probs: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
        prop_bin = in_bin.mean()
        
        if prop_bin > 0:
            avg_pred_prob = pred_probs[in_bin].mean()
            avg_true_prob = true_probs[in_bin].mean()
            ece += np.abs(avg_pred_prob - avg_true_prob) * prop_bin
    
    return float(ece)

def model_metrics(clf_preds: torch.Tensor, clf_labels: torch.Tensor, clf_probs: torch.Tensor, total_loss_clf: float, n_samples: int, risk_names: list = None) -> Dict[str, Any]:
    if risk_names is None:
        risk_names = RISK_NAMES
    
    metrics = {}
    metrics["loss_clf"] = total_loss_clf / n_samples
    n_targets = clf_preds.shape[1]
    
    for i in range(n_targets):
        risk_pred = clf_preds[:, i]
        risk_label = clf_labels[:, i].int()
        risk_prob = clf_probs[:, i]
        
        risk_label_np = risk_label.cpu().numpy()
        risk_prob_np = risk_prob.cpu().numpy()
        
        metrics[f"accuracy_risk_{risk_names[i]}"] = float(accuracy(risk_pred, risk_label, task="binary").cpu())
        metrics[f"precision_risk_{risk_names[i]}"] = float(precision(risk_pred, risk_label, task="binary").cpu())
        metrics[f"recall_risk_{risk_names[i]}"] = float(recall(risk_pred, risk_label, task="binary").cpu())
        metrics[f"f1_risk_{risk_names[i]}"] = float(f1_score(risk_pred, risk_label, task="binary").cpu())
        metrics[f"mcc_risk_{risk_names[i]}"] = matthews_corrcoef(risk_label_np, risk_pred.cpu().numpy())
        
        metrics[f"roc_auc_risk_{risk_names[i]}"] = float(auroc(risk_prob, risk_label, task="binary").cpu())
        
        if len(np.unique(risk_label_np)) > 1:
            metrics[f"brier_score_risk_{risk_names[i]}"] = brier_score_loss(risk_label_np, risk_prob_np)
        else:
            metrics[f"brier_score_risk_{risk_names[i]}"] = np.nan
        
        if len(np.unique(risk_label_np)) > 1:
            metrics[f"pr_auc_risk_{risk_names[i]}"] = average_precision_score(risk_label_np, risk_prob_np)
        else:
            metrics[f"pr_auc_risk_{risk_names[i]}"] = np.nan
        
        ece_value = ece(risk_prob_np, risk_label_np, n_bins=10)
        metrics[f"ece_risk_{risk_names[i]}"] = ece_value
    
    metrics["accuracy_clf_macro"] = sum([metrics[f"accuracy_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
    metrics["precision_clf_macro"] = sum([metrics[f"precision_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
    metrics["recall_clf_macro"] = sum([metrics[f"recall_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
    metrics["f1_clf_macro"] = sum([metrics[f"f1_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
    metrics["roc_auc_clf_macro"] = sum([metrics[f"roc_auc_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
    
    brier_scores = [metrics[f"brier_score_risk_{risk_names[i]}"] for i in range(n_targets) if not np.isnan(metrics[f"brier_score_risk_{risk_names[i]}"])]
    ece_scores = [metrics[f"ece_risk_{risk_names[i]}"] for i in range(n_targets)]
    pr_auc_scores = [metrics[f"pr_auc_risk_{risk_names[i]}"] for i in range(n_targets) if not np.isnan(metrics[f"pr_auc_risk_{risk_names[i]}"])]
    
    if brier_scores:
        metrics["brier_score_macro"] = np.mean(brier_scores)
    if ece_scores:
        metrics["ece_macro"] = np.mean(ece_scores)
    if pr_auc_scores:
        metrics["pr_auc_macro"] = np.mean(pr_auc_scores)
    
    return metrics

def dataset_metrics(data: Dict[str, Any], y_train, y_test, risk_names: list = None) -> Dict[str, Any]:
    if risk_names is None:
        risk_names = RISK_NAMES
    
    n_features = data['train']['X'].shape[1]
    
    dataset_metrics = {
        "dataset/train_n_samples": len(data['train']['X']),
        "dataset/val_n_samples": len(data['val']['X']),
        "dataset/test_n_samples": len(data['test']['X']),
        "dataset/train_n_features": n_features,
    }
    
    for i, risk_name in enumerate(risk_names):
        risk_col = f"Risk_{risk_name}"
        if risk_col in y_train.columns:
            pos_count = y_train[risk_col].sum()
            pos_ratio = pos_count / len(y_train)
            dataset_metrics[f"dataset/train_{risk_col}_positive_count"] = int(pos_count)
            dataset_metrics[f"dataset/train_{risk_col}_positive_ratio"] = float(pos_ratio)
    
    for i, risk_name in enumerate(risk_names):
        risk_col = f"Risk_{risk_name}"
        if risk_col in y_test.columns:
            pos_count = y_test[risk_col].sum()
            pos_ratio = pos_count / len(y_test)
            dataset_metrics[f"dataset/test_{risk_col}_positive_count"] = int(pos_count)
            dataset_metrics[f"dataset/test_{risk_col}_positive_ratio"] = float(pos_ratio)
    
    return dataset_metrics

def model_metrics_probability(clf_probs: torch.Tensor, clf_true_probs: torch.Tensor, risk_names: list = None) -> Dict[str, Any]:
    if risk_names is None:
        risk_names = RISK_NAMES
    
    metrics = {}
    n_targets = clf_probs.shape[1]
    
    for i in range(n_targets):
        pred_prob = clf_probs[:, i]
        true_prob = clf_true_probs[:, i]
        
        pred_prob_np = pred_prob.cpu().numpy()
        true_prob_np = true_prob.cpu().numpy()
        
        pred_prob_np = np.clip(pred_prob_np, 0.0, 1.0)
        true_prob_np = np.clip(true_prob_np, 0.0, 1.0)
        
        mse = np.mean((pred_prob_np - true_prob_np) ** 2)
        metrics[f"mse_risk_{risk_names[i]}"] = float(mse)
        
        mae = np.mean(np.abs(pred_prob_np - true_prob_np))
        metrics[f"mae_risk_{risk_names[i]}"] = float(mae)
        
        ece_value = ece_probability(pred_prob_np, true_prob_np, n_bins=10)
        metrics[f"ece_prob_risk_{risk_names[i]}"] = ece_value
    
    return metrics

def model_metrics_categories(pred_categories: np.ndarray, true_categories: np.ndarray, risk_names: list = None, prefix: str = "category") -> Dict[str, Any]:
    if risk_names is None:
        risk_names = RISK_NAMES
    
    metrics = {}
    n_targets = pred_categories.shape[1] if len(pred_categories.shape) > 1 else 1
    
    if len(pred_categories.shape) == 1:
        pred_categories = pred_categories.reshape(-1, 1)
        true_categories = true_categories.reshape(-1, 1)
        risk_names = [risk_names[0]] if isinstance(risk_names, list) else [risk_names]
    
    for i in range(n_targets):
        pred_cat = pred_categories[:, i]
        true_cat = true_categories[:, i]
        
        acc = (pred_cat == true_cat).mean()
        metrics[f"{prefix}_accuracy_risk_{risk_names[i]}"] = float(acc)
        
        f1_macro = sk_f1_score(true_cat, pred_cat, average='macro', zero_division=0)
        metrics[f"{prefix}_f1_macro_risk_{risk_names[i]}"] = float(f1_macro)
    
    return metrics

def compute_consistency_metrics(categorizations: Dict[Any, np.ndarray],  prefix: str = "consistency", risk_names: list = None, client_ids: list = None) -> Dict[str, Any]:
    if risk_names is None:
        risk_names = RISK_NAMES
    
    if client_ids is None:
        client_ids = sorted(categorizations.keys())
    
    if len(client_ids) < 2:
        return {}
    
    metrics = {}
    n_risks = len(risk_names)
    
    for risk_idx, risk_name in enumerate(risk_names):
        risk_categories_list = []
        for client_id in client_ids:
            if client_id in categorizations:
                cat_list = categorizations[client_id]
                if len(cat_list.shape) == 2:
                    risk_categories_list.append(cat_list[:, risk_idx])
                else:
                    risk_categories_list.append(cat_list)
        
        if len(risk_categories_list) < 2:
            continue
        
        risk_categories = np.stack(risk_categories_list, axis=1)
        
        # for global sanity check (should be 0.0), just compare first column with all others and exit early
        all_identical = True
        if risk_categories.shape[1] > 1:
            first_col = risk_categories[:, 0]
            for col_idx in range(1, risk_categories.shape[1]):
                if not np.array_equal(risk_categories[:, col_idx], first_col):
                    all_identical = False
                    break
            
            if all_identical:
                inconsistency_any_score = 0.0
                inconsistency_dist_score = 0.0
                patient_disagree_score = 0.0
            else:
                inconsistency_any_score = inconsistency_any(risk_categories)
                inconsistency_dist_score = inconsistency_distance(risk_categories)
                patient_disagree_score = patient_disagreement(risk_categories)
        else:
            inconsistency_any_score = inconsistency_any(risk_categories)
            inconsistency_dist_score = inconsistency_distance(risk_categories)
            patient_disagree_score = patient_disagreement(risk_categories)
        
        metrics[f"{prefix}/inconsistency_any_risk_{risk_name}"] = float(inconsistency_any_score)
        metrics[f"{prefix}/inconsistency_distance_risk_{risk_name}"] = float(inconsistency_dist_score)
        metrics[f"{prefix}/patient_disagreement_risk_{risk_name}"] = float(patient_disagree_score)
    
    if metrics:
        inconsistency_any_scores = [metrics[f"{prefix}/inconsistency_any_risk_{risk_name}"] for risk_name in risk_names if f"{prefix}/inconsistency_any_risk_{risk_name}" in metrics]
        inconsistency_dist_scores = [metrics[f"{prefix}/inconsistency_distance_risk_{risk_name}"] for risk_name in risk_names if f"{prefix}/inconsistency_distance_risk_{risk_name}" in metrics]
        patient_disagree_scores = [metrics[f"{prefix}/patient_disagreement_risk_{risk_name}"] for risk_name in risk_names if f"{prefix}/patient_disagreement_risk_{risk_name}" in metrics]
        
        if inconsistency_any_scores:
            metrics[f"{prefix}/inconsistency_any_macro"] = float(np.mean(inconsistency_any_scores))
            metrics[f"{prefix}/inconsistency_distance_macro"] = float(np.mean(inconsistency_dist_scores))
            metrics[f"{prefix}/patient_disagreement_macro"] = float(np.mean(patient_disagree_scores))
    
    return metrics