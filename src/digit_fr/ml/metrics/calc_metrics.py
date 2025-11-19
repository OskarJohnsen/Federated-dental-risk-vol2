from typing import Dict, Any
import torch
import numpy as np
from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc
from sklearn.metrics import matthews_corrcoef, brier_score_loss, average_precision_score
from ..constants import RISK_NAMES

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