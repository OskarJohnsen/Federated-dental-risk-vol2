import wandb
import torch
from torch.optim.lr_scheduler import StepLR
from ..models.base.trainer import BaseTrainer
from ...core.paths import root_path
from ..models.architectures.mlp import MLP
from ..data.loaders import load_data_with_split, load_global_test_set
from ..data.datasets import create_data_loaders
from ..constants import RISK_NAMES
from ..metrics.calc_metrics import dataset_metrics, model_metrics_categories, compute_consistency_metrics
from ..metrics.threshold import percentile_thresholds, apply_risk_categorization, load_global_thresholds
from ..metrics.report import log_metrics_wandb, log_dataset_info, log_experiment_config
import pandas as pd
from ..util.seed import all_seeds
from ..config.experiment_config import ExperimentConfig, get_data_version
from typing import Optional
import numpy as np

def main(config: ExperimentConfig):
    all_seeds(config.model_seed)

    wandb.init(
        project="digit-federated-recommenders",
        name=config.get_run_name(),
        config=config.to_wandb_config(),
        tags=[config.experiment_type, config.model, f"seed{config.model_seed}", config.threshold_method],
        group=config.get_group_name(),
        job_type=config.experiment_type,
    )
    
    if config.dataset_path is None:
        config.dataset_path = str(root_path('data', 'raw', 'fed_recommenders_synthetic_dataset_50k.csv'))
    if config.test_set_path is None:
        config.test_set_path = str(root_path('data', 'processed', 'global_test_set.csv'))
    
    if config.data_version is None:
        config.data_version = get_data_version(config.dataset_path)
    
    print(f"{config.experiment_type.upper()} TRAINING")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Seed (Model): {config.model_seed}, Seed (Data): {config.data_split_seed}")

    print("\nData")
    data = load_data_with_split(config=config)
    n_features = data['train']['X'].shape[1]
    print(f"Features: {n_features}")
    print(f"Train samples: {len(data['train']['X']):,}")
    print(f"Val samples: {len(data['val']['X']):,}")
    
    preprocessing_pipeline = data.get('_preprocessing_pipeline')
    if preprocessing_pipeline is None:
        raise ValueError("Preprocessing pipeline not found in training data.")
    
    global_test_data = load_global_test_set(test_set_path=config.test_set_path, preprocessing_pipeline=preprocessing_pipeline)
    print(f"Global test samples: {len(global_test_data['X']):,}")
    if 'y_probabilities' in global_test_data:
        print("Probability labels exist in global test")
    if 'y_categories' in global_test_data:
        print("Category labels exist in global test")
    
    global_test_X = global_test_data['X']
    
    missing_cols = set(data['train']['X'].columns) - set(global_test_X.columns)
    extra_cols = set(global_test_X.columns) - set(data['train']['X'].columns)
    
    if missing_cols:
        print(f"STOP: Test set missing {len(missing_cols)} columns from training.")
    if extra_cols:
        print(f"STOP: Test set has {len(extra_cols)} extra columns.")
    
    global_test_X = global_test_X.reindex(columns=data['train']['X'].columns, fill_value=0.0)
    
    data['test'] = {
        'X': global_test_X,
        'y_classification': global_test_data['y_classification']
    }
    if 'y_probabilities' in global_test_data:
        data['test']['y_probabilities'] = global_test_data['y_probabilities']
    if 'y_categories' in global_test_data:
        data['test']['y_categories'] = global_test_data['y_categories']
    if 'Client' in global_test_data:
        data['test']['Client'] = global_test_data['Client']
    
    y_train = data["train"]["y_classification"]
    y_val = data["val"]["y_classification"]
    y_test = data["test"]["y_classification"]
    
    dataset_metrics_log = dataset_metrics(data, y_train, y_test, RISK_NAMES)
    log_dataset_info(dataset_metrics_log, RISK_NAMES)
    
    train_loader, val_loader, test_loader = create_data_loaders(data, batch_size=config.batch_size)
    
    model = MLP(
        input_size=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        n_clf_classes=4,
    )
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_weights = None
    if config.use_class_weights:
        pos_counts = y_train.sum(axis=0).values.astype(float)
        total = len(y_train)
        neg_counts = total - pos_counts
        eps = 1e-6
        max_weight = 100.0
        pos_weights = torch.clamp(torch.tensor(neg_counts / (pos_counts + eps), dtype=torch.float32), min=1.0, max=max_weight)
        loss_clf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        loss_clf = torch.nn.BCEWithLogitsLoss()
    
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Not a optimizer in config: {config.optimizer}")
    
    scheduler = None
    if config.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        loss_clf=loss_clf,
        scheduler=scheduler,
        experiment_type=config.experiment_type,
        seed=config.model_seed
    )
    
    print(f"Training on device: {trainer.device}")
    config.input_size = n_features
    log_experiment_config(config, model, pos_weights)
    
    results = trainer.fit(train_loader, val_loader=val_loader, epochs=config.epochs)

    for epoch, (train_loss, val_loss) in enumerate(zip(results['history']['train_loss'], results['history']['val_loss']), 1):
        epoch_metrics = {
            "train/loss": train_loss,
            "val/loss": val_loss,
        }
        log_metrics_wandb(epoch_metrics, prefix="", epoch=epoch)

    print("EVAL GLOBAL TEST SET")
    test_metrics = trainer.evaluate(test_loader, thresholds=None)
    test_probs = test_metrics.get("_probs")
    test_true_categories = test_metrics.get("_true_categories")
    
    all_test_metrics = {}
    
    if "_true_probs" in test_metrics:
        print("PROB METRICS")
        for risk_name in RISK_NAMES:
            mse_key = f"mse_risk_{risk_name}"
            mae_key = f"mae_risk_{risk_name}"
            if mse_key in test_metrics:
                print(f"{risk_name}: MSE={test_metrics[mse_key]}, MAE={test_metrics[mae_key]}")
        if "mse_macro" in test_metrics:
            print(f"Macro: MSE={test_metrics['mse_macro']}, MAE={test_metrics['mae_macro']}")
        all_test_metrics.update({k: v for k, v in test_metrics.items() if k.startswith(('mse_', 'mae_', 'brier_score_prob_', 'ece_prob_'))})
    
    global_thresholds = load_global_thresholds()
    
    val_client_ids = data['val']['Client'].values
    unique_clients = np.unique(val_client_ids)
    
    print(f"\nFound {len(unique_clients)} clients in validation set")
    
    if config.category_strategy in ["per_client", "both"]:
        print("PER-CLIENT CATEGORY EVAL")
        val_metrics = trainer.evaluate(val_loader, thresholds=None)
        val_probs = val_metrics.get("_probs")
        val_client_ids_arr = np.array(val_client_ids)
        
        per_client_thresholds = {}
        for client_id in unique_clients:
            client_mask = val_client_ids_arr == client_id
            if client_mask.sum() > 0:
                client_val_probs = val_probs[client_mask]
                thresholds = percentile_thresholds(client_val_probs, percentiles=[33, 67], risk_names=RISK_NAMES)
                per_client_thresholds[client_id] = thresholds
                print(f"Client {client_id}: {client_mask.sum()} validation samples")
        
        test_client_ids = data['test'].get('Client')
        if test_client_ids is not None:
            if isinstance(test_client_ids, pd.Series):
                test_client_ids_arr = test_client_ids.values
            else:
                test_client_ids_arr = np.array(test_client_ids)
            
            all_per_client_pred_categories = []
            all_per_client_true_categories = []
            client_indices_for_metrics = []
            
            for client_id in unique_clients:
                if client_id in per_client_thresholds:
                    client_test_mask = test_client_ids_arr == client_id
                    if client_test_mask.sum() > 0:
                        client_test_probs = test_probs[client_test_mask]
                        client_true_cats = test_true_categories[client_test_mask] if test_true_categories is not None else None
                        
                        pred_categories = apply_risk_categorization(client_test_probs, per_client_thresholds[client_id], risk_names=RISK_NAMES)
                        
                        all_per_client_pred_categories.append(pred_categories)
                        if client_true_cats is not None:
                            all_per_client_true_categories.append(client_true_cats)
                        client_indices_for_metrics.append(client_id)
            
            if all_per_client_pred_categories:
                combined_pred_cats = np.vstack(all_per_client_pred_categories)
                combined_true_cats = np.vstack(all_per_client_true_categories) if all_per_client_true_categories else None
                
                if combined_true_cats is not None:
                    per_client_cat_metrics = model_metrics_categories(combined_pred_cats, combined_true_cats, risk_names=RISK_NAMES, prefix="category_per_client")
                    all_test_metrics.update(per_client_cat_metrics)
                    print(f"Per-Client category metrics done")
                
                # Per-Client Consistency
                if len(per_client_thresholds) > 1:
                    print("\nCONSISTENCY METRICS: Per-Client Thresholds")
                    
                    per_client_categorizations_full = {}
                    for client_id in per_client_thresholds.keys():
                        client_categories = apply_risk_categorization(test_probs, per_client_thresholds[client_id], risk_names=RISK_NAMES)
                        per_client_categorizations_full[client_id] = client_categories
                    
                    consistency_metrics_per_client = compute_consistency_metrics(categorizations=per_client_categorizations_full, prefix="consistency_per_client", risk_names=RISK_NAMES, client_ids=sorted(per_client_thresholds.keys()))
                    
                    if consistency_metrics_per_client:
                        for risk_name in RISK_NAMES:
                            any_key = f"consistency_per_client/inconsistency_any_risk_{risk_name}"
                            dist_key = f"consistency_per_client/inconsistency_distance_risk_{risk_name}"
                            disagree_key = f"consistency_per_client/patient_disagreement_risk_{risk_name}"
                            
                            if any_key in consistency_metrics_per_client:
                                print(f"\n{risk_name}:")
                                print(f"Inconsistency (any): {consistency_metrics_per_client[any_key]}")
                                print(f"Inconsistency (distance): {consistency_metrics_per_client[dist_key]}")
                                print(f"Patient disagreement: {consistency_metrics_per_client[disagree_key]}")
                        
                        if "consistency_per_client/inconsistency_any_macro" in consistency_metrics_per_client:
                            print(f"\nMacro Averages (Per-Client Thresholds):")
                            print(f"Inconsistency (any): {consistency_metrics_per_client['consistency_per_client/inconsistency_any_macro']}")
                            print(f"Inconsistency (distance): {consistency_metrics_per_client['consistency_per_client/inconsistency_distance_macro']}")
                            print(f"Patient disagreement: {consistency_metrics_per_client['consistency_per_client/patient_disagreement_macro']}")
                        
                        all_test_metrics.update(consistency_metrics_per_client)
    
    if config.category_strategy in ["global", "both"]:
        print("GLOBAL CATEGORY EVAL")
        
        global_pred_categories = apply_risk_categorization(test_probs, global_thresholds, risk_names=RISK_NAMES)
        
        if test_true_categories is not None:
            global_cat_metrics = model_metrics_categories(global_pred_categories, test_true_categories, risk_names=RISK_NAMES, prefix="category_global")
            all_test_metrics.update(global_cat_metrics)
            for risk_name in RISK_NAMES:
                acc_key = f"category_global_accuracy_risk_{risk_name}"
                f1_key = f"category_global_f1_macro_risk_{risk_name}"
                if acc_key in global_cat_metrics:
                    print(f"{risk_name}: Accuracy={global_cat_metrics[acc_key]}, F1-macro={global_cat_metrics[f1_key]}")
        
        test_client_ids = data['test'].get('Client')
        if test_client_ids is not None:
            if isinstance(test_client_ids, pd.Series):
                test_client_ids_arr = test_client_ids.values
            else:
                test_client_ids_arr = np.array(test_client_ids)
            
            unique_test_clients = np.unique(test_client_ids_arr)
            
            if len(unique_test_clients) > 1:
                print("\nCONSISTENCY METRICS: Global Thresholds")
                print("(Sanity check: Should be 0.0)")
                
                n_test_samples = len(global_pred_categories)
                n_clients = len(unique_test_clients)
                global_categorizations_sanity = {}
                for client_id in sorted(unique_test_clients):
                    global_categorizations_sanity[client_id] = global_pred_categories
                
                consistency_metrics_global = compute_consistency_metrics(
                    categorizations=global_categorizations_sanity,
                    prefix="consistency_global",
                    risk_names=RISK_NAMES,
                    client_ids=sorted(unique_test_clients)
                )
                
                if consistency_metrics_global:
                    for risk_name in RISK_NAMES:
                        any_key = f"consistency_global/inconsistency_any_risk_{risk_name}"
                        dist_key = f"consistency_global/inconsistency_distance_risk_{risk_name}"
                        disagree_key = f"consistency_global/patient_disagreement_risk_{risk_name}"
                        
                        if any_key in consistency_metrics_global:
                            print(f"\n{risk_name}:")
                            print(f"Inconsistency (any): {consistency_metrics_global[any_key]} (expected: 0.0)")
                            print(f"Inconsistency (distance): {consistency_metrics_global[dist_key]} (expected: 0.0)")
                            print(f"Patient disagreement: {consistency_metrics_global[disagree_key]} (expected: 0.0)")
                    
                    if "consistency_global/inconsistency_any_macro" in consistency_metrics_global:
                        print(f"\nMacro Averages (Global Thresholds):")
                        print(f"Inconsistency (any): {consistency_metrics_global['consistency_global/inconsistency_any_macro']} (expected: 0.0)")
                        print(f"Inconsistency (distance): {consistency_metrics_global['consistency_global/inconsistency_distance_macro']} (expected: 0.0)")
                        print(f"Patient disagreement: {consistency_metrics_global['consistency_global/patient_disagreement_macro']} (expected: 0.0)")
                    
                    all_test_metrics.update(consistency_metrics_global)
    
    log_metrics_wandb(all_test_metrics, prefix="test/")
    
    print("\nDone")
    wandb.finish()

if __name__ == '__main__':
    main()