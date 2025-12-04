import wandb
import torch
import copy
from torch.optim.lr_scheduler import StepLR
from ..models.base.trainer import BaseTrainer
from ...core.paths import root_path
from ..models.architectures.mlp import MLP
from ..data.loaders import load_raw_data, load_data_per_client, load_global_test_set
from ..data.datasets import create_data_loaders, MultiTaskDataset
from ..data.preprocessing import PreprocessingPipeline
from torch.utils.data import DataLoader
from ..constants import RISK_NAMES
from sklearn.model_selection import train_test_split
from .aggregation import federated_averaging
from ..metrics.calc_metrics import dataset_metrics, model_metrics_categories, compute_consistency_metrics
from ..metrics.threshold import percentile_thresholds, apply_risk_categorization, load_global_thresholds
from ..metrics.report import log_metrics_wandb, log_dataset_info, log_experiment_config
import pandas as pd
from ..util.seed import all_seeds, data_seeds
from ..config.experiment_config import ExperimentConfig, get_data_version
from typing import Optional, Tuple
import numpy as np
import random


def load_client_data_with_global_pipeline(full_data: dict, client_id: int, global_pipeline: PreprocessingPipeline, config: ExperimentConfig) -> dict:
    data_seeds(config.data_split_seed)
    client_mask = full_data['Client'] == client_id
    X_client = full_data['X'][client_mask].copy()
    y_client = full_data['y_classification'][client_mask].copy()
    
    if len(X_client) == 0:
        raise ValueError(f"No data found for client {client_id}")
    
    val_size_adjusted = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_client, y_client, test_size=val_size_adjusted, random_state=config.data_split_seed)
    
    X_train_processed = global_pipeline.transform(X_train)
    X_val_processed = global_pipeline.transform(X_val)
    
    result = {
        'train': {'X': X_train_processed, 'y_classification': y_train},
        'val': {'X': X_val_processed, 'y_classification': y_val},
        '_preprocessing_pipeline': global_pipeline
    }
    
    return result


def train_client_locally(model: torch.nn.Module, client_data: dict, config: ExperimentConfig, local_epochs: int) -> Tuple[torch.nn.Module, int]:
    y_train = client_data["train"]["y_classification"]
    n_samples = len(y_train)
    
    pos_weights = None
    if config.use_class_weights:
        pos_counts = y_train.sum(axis=0).values.astype(float)
        total = len(y_train)
        neg_counts = total - pos_counts
        eps = 1e-6
        max_weight = 100.0
        pos_weights = torch.clamp( torch.tensor(neg_counts / (pos_counts + eps), dtype=torch.float32), min=1.0, max=max_weight)
        loss_clf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        loss_clf = torch.nn.BCEWithLogitsLoss()
    
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    scheduler = None
    if config.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    
    train_loader, val_loader = create_data_loaders(client_data, batch_size=config.batch_size)
    
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        loss_clf=loss_clf,
        scheduler=scheduler,
        experiment_type=config.experiment_type,
        seed=config.model_seed
    )
    
    trainer.fit(train_loader, val_loader=val_loader, epochs=local_epochs)
    
    return model, n_samples


def main(config: ExperimentConfig):
    all_seeds(config.model_seed)
    
    if config.federated_rounds is None:
        raise ValueError("federated_rounds not provided in config")
    if config.local_epochs is None:
        raise ValueError("local_epochs not provided in config")
    
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
    print(f"Federated Rounds: {config.federated_rounds}, Local Epochs: {config.local_epochs}")
    if config.clients_per_round:
        print(f"Clients per round: {config.clients_per_round}")
    else:
        print(f"Using all clients per round")
    
    print("\nLoading full dataset...")
    full_data = load_raw_data(dataset_path=config.dataset_path)
    client_ids = sorted(full_data['Client'].unique())
    print(f"Found {len(client_ids)} clients: {client_ids}")
    
    print("\nCreating global preprocessing pipeline from all training data...")
    all_training_data = []
    for client_id in client_ids:
        client_mask = full_data['Client'] == client_id
        X_client = full_data['X'][client_mask].copy()
        val_size_adjusted = config.val_size / (1 - config.test_size)
        data_seeds(config.data_split_seed)
        X_train, _, _, _ = train_test_split(X_client, full_data['y_classification'][client_mask], test_size=val_size_adjusted, random_state=config.data_split_seed)
        all_training_data.append(X_train)
    
    combined_training_data = pd.concat(all_training_data, ignore_index=True)
    global_preprocessing_pipeline = PreprocessingPipeline()
    global_preprocessing_pipeline.fit(combined_training_data)
    print(f"Global preprocessing pipeline fitted on {len(combined_training_data):,} samples")
    
    n_features = len(global_preprocessing_pipeline.feature_columns)
    print(f"Features: {n_features}")
    
    global_test_data = load_global_test_set(test_set_path=config.test_set_path, preprocessing_pipeline=global_preprocessing_pipeline)
    print(f"Global test samples: {len(global_test_data['X']):,}")
    
    global_test_X = global_test_data['X'].copy()
    
    global_test_X = global_test_X.reindex(columns=global_preprocessing_pipeline.feature_columns, fill_value=0.0)
    
    test_data = {
        'X': global_test_X,
        'y_classification': global_test_data['y_classification']
    }
    if 'y_probabilities' in global_test_data:
        test_data['y_probabilities'] = global_test_data['y_probabilities']
    if 'y_categories' in global_test_data:
        test_data['y_categories'] = global_test_data['y_categories']
    if 'Client' in global_test_data:
        test_data['Client'] = global_test_data['Client']
    
    y_test_probs = test_data.get('y_probabilities', None)
    y_test_categories = test_data.get('y_categories', None)
    test_dataset = MultiTaskDataset(test_data['X'],  test_data['y_classification'],  y_probabilities=y_test_probs,  y_categories=y_test_categories)
    test_size = len(test_dataset)
    test_batch_size = min(config.batch_size, test_size) if test_size > 0 else config.batch_size
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)
    
    global_thresholds = load_global_thresholds()
    
    global_model = MLP(
        input_size=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        n_clf_classes=4,
    )
    print(f"\nGlobal Model: {global_model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    
    config.input_size = n_features
    log_experiment_config(config, global_model, None)
    
    if config.clients_per_round:
        random.seed(config.model_seed)
    
    for round_num in range(config.federated_rounds):
        print(f"\n{'='*60}")
        print(f"Federated Round {round_num + 1}/{config.federated_rounds}")
        print(f"{'='*60}")
        
        if config.clients_per_round:
            selected_clients = random.sample(client_ids, min(config.clients_per_round, len(client_ids)))
        else:
            selected_clients = client_ids
        
        print(f"Selected clients: {selected_clients}")
        
        client_weights = []
        client_sample_counts = []
        
        for client_id in selected_clients:
            print(f"\nTraining Client {client_id}...")
            client_data = load_client_data_with_global_pipeline(full_data, client_id, global_preprocessing_pipeline, config)

            client_model = copy.deepcopy(global_model)
            
            client_model, n_samples = train_client_locally(client_model, client_data, config, config.local_epochs)
            
            client_weights.append(client_model.state_dict())
            client_sample_counts.append(n_samples)
            print(f"Client {client_id}: {n_samples:,} training samples")
        
        print(f"\nAggregating weights from {len(selected_clients)} clients...")
        aggregated_weights = federated_averaging(client_weights, client_sample_counts)
        
        global_model.load_state_dict(aggregated_weights)
        
        print(f"Evaluating global model on test set...")
        trainer = BaseTrainer(
            model=global_model,
            optimizer=None,
            loss_clf=torch.nn.BCEWithLogitsLoss(),
            scheduler=None,
            experiment_type=config.experiment_type,
            seed=config.model_seed
        )
        
        test_metrics = trainer.evaluate(test_loader, thresholds=None)
        
        round_metrics = {}
        prob_keys = [k for k in test_metrics.keys() if k.startswith(('mse_', 'mae_', 'brier_score_prob_', 'ece_prob_'))]
        for key in prob_keys:
            round_metrics[f"round_{round_num + 1}/test/{key}"] = test_metrics[key]
        
        if 'loss_clf' in test_metrics:
            round_metrics[f"round_{round_num + 1}/test/loss"] = test_metrics['loss_clf']
        
        log_metrics_wandb(round_metrics, prefix="")
        
        if "mse_macro" in test_metrics:
            print(f"  Round {round_num + 1} - Test MSE (macro): {test_metrics['mse_macro']:.6f}")
    
    print("EVALUATION")
    
    final_test_metrics = trainer.evaluate(test_loader, thresholds=None)
    test_probs = final_test_metrics.get("_probs")
    test_true_categories = final_test_metrics.get("_true_categories")
    
    all_test_metrics = {}
    
    if "_true_probs" in final_test_metrics:
        print("\nPROBABILITY METRICS")
        for risk_name in RISK_NAMES:
            mse_key = f"mse_risk_{risk_name}"
            mae_key = f"mae_risk_{risk_name}"
            if mse_key in final_test_metrics:
                print(f"{risk_name}: MSE={final_test_metrics[mse_key]:.6f}, MAE={final_test_metrics[mae_key]:.6f}")
        if "mse_macro" in final_test_metrics:
            print(f"Macro: MSE={final_test_metrics['mse_macro']:.6f}, MAE={final_test_metrics['mae_macro']:.6f}")
        all_test_metrics.update({k: v for k, v in final_test_metrics.items() if k.startswith(('mse_', 'mae_', 'brier_score_prob_', 'ece_prob_'))})
    
    if config.category_strategy in ["per_client", "both"] and test_probs is not None:
        print("\nPER-CLIENT CATEGORY EVAL")
        
        per_client_thresholds = {}
        unique_clients = []
        
        for client_id in client_ids:
            client_data = load_client_data_with_global_pipeline(full_data, client_id, global_preprocessing_pipeline, config)
            
            if 'val' not in client_data or len(client_data['val']['X']) == 0:
                print(f"  Skipping client {client_id}: no validation data")
                continue
            
            _, val_loader = create_data_loaders(client_data, batch_size=config.batch_size)
            val_metrics = trainer.evaluate(val_loader, thresholds=None)
            val_probs = val_metrics.get("_probs")
            
            if val_probs is not None and len(val_probs) > 0:
                thresholds = percentile_thresholds(val_probs=val_probs, percentiles=[33, 67], risk_names=RISK_NAMES)
                per_client_thresholds[client_id] = thresholds
                unique_clients.append(client_id)
                print(f"  Client {client_id}: {len(val_probs)} validation samples")
        
        if per_client_thresholds:
            if test_true_categories is not None:
                all_per_client_pred_categories = []
                
                for client_id in unique_clients:
                    if client_id in per_client_thresholds:
                        pred_categories = apply_risk_categorization(test_probs, per_client_thresholds[client_id], risk_names=RISK_NAMES)
                        
                        per_client_cat_metrics = model_metrics_categories(
                            pred_categories, 
                            test_true_categories, 
                            risk_names=RISK_NAMES, 
                            prefix=f"category_per_client_client_{client_id}"
                        )
                        all_test_metrics.update(per_client_cat_metrics)
                        
                        all_per_client_pred_categories.append(pred_categories)
                
                if all_per_client_pred_categories:
                    combined_pred_cats = np.vstack(all_per_client_pred_categories)
                    combined_true_cats = np.vstack([test_true_categories] * len(all_per_client_pred_categories))
                    per_client_cat_metrics_macro = model_metrics_categories(combined_pred_cats, combined_true_cats, risk_names=RISK_NAMES, prefix="category_per_client")
                    all_test_metrics.update(per_client_cat_metrics_macro)
                    print(f"Per-Client category metrics done")
                    
                    for risk_name in RISK_NAMES:
                        acc_key = f"category_per_client_accuracy_risk_{risk_name}"
                        f1_key = f"category_per_client_f1_macro_risk_{risk_name}"
                        if acc_key in per_client_cat_metrics_macro:
                            print(f"{risk_name}: Accuracy={per_client_cat_metrics_macro[acc_key]:.4f}, F1-macro={per_client_cat_metrics_macro[f1_key]:.4f}")
                    
                    if len(per_client_thresholds) > 1:
                        print("\nCONSISTENCY METRICS: Per-Client Thresholds")
                        
                        per_client_categorizations_full = {}
                        for client_id in per_client_thresholds.keys():
                            client_categories = apply_risk_categorization(test_probs, per_client_thresholds[client_id], risk_names=RISK_NAMES)
                            per_client_categorizations_full[client_id] = client_categories
                        
                        if len(per_client_categorizations_full) > 1:
                            consistency_metrics_per_client = compute_consistency_metrics(categorizations=per_client_categorizations_full, prefix="consistency_per_client", risk_names=RISK_NAMES, client_ids=sorted(per_client_categorizations_full.keys()))
                            
                            if consistency_metrics_per_client:
                                for risk_name in RISK_NAMES:
                                    any_key = f"consistency_per_client/inconsistency_any_risk_{risk_name}"
                                    dist_key = f"consistency_per_client/inconsistency_distance_risk_{risk_name}"
                                    disagree_key = f"consistency_per_client/patient_disagreement_risk_{risk_name}"
                                    
                                    if any_key in consistency_metrics_per_client:
                                        print(f"\n{risk_name}:")
                                        print(f"Inconsistency (any): {consistency_metrics_per_client[any_key]:.6f}")
                                        print(f"Inconsistency (distance): {consistency_metrics_per_client[dist_key]:.6f}")
                                        print(f"Patient disagreement: {consistency_metrics_per_client[disagree_key]:.6f}")
                                
                                if "consistency_per_client/inconsistency_any_macro" in consistency_metrics_per_client:
                                    print(f"\nMacro Averages (Per-Client Thresholds):")
                                    print(f"Inconsistency (any): {consistency_metrics_per_client['consistency_per_client/inconsistency_any_macro']:.6f}")
                                    print(f"Inconsistency (distance): {consistency_metrics_per_client['consistency_per_client/inconsistency_distance_macro']:.6f}")
                                    print(f"Patient disagreement: {consistency_metrics_per_client['consistency_per_client/patient_disagreement_macro']:.6f}")
                                
                                all_test_metrics.update(consistency_metrics_per_client)
    
    if config.category_strategy in ["global", "both"] and test_probs is not None:
        print("\nGLOBAL CATEGORY EVAL")
        
        global_pred_categories = apply_risk_categorization(test_probs, global_thresholds, risk_names=RISK_NAMES)
        
        if test_true_categories is not None:
            global_cat_metrics = model_metrics_categories(global_pred_categories, test_true_categories, risk_names=RISK_NAMES, prefix="category_global")
            all_test_metrics.update(global_cat_metrics)
            for risk_name in RISK_NAMES:
                acc_key = f"category_global_accuracy_risk_{risk_name}"
                f1_key = f"category_global_f1_macro_risk_{risk_name}"
                if acc_key in global_cat_metrics:
                    print(f"{risk_name}: Accuracy={global_cat_metrics[acc_key]:.4f}, F1-macro={global_cat_metrics[f1_key]:.4f}")
        
        test_client_ids = test_data.get('Client')
        if test_client_ids is not None:
            if isinstance(test_client_ids, pd.Series):
                test_client_ids_arr = test_client_ids.values
            else:
                test_client_ids_arr = np.array(test_client_ids)
            
            unique_test_clients = np.unique(test_client_ids_arr)
            
            if len(unique_test_clients) > 1:
                print("\nCONSISTENCY METRICS: Global Thresholds")
                print("(Sanity check: Should be 0.0)")
                
                global_categorizations_sanity = {}
                for client_id in sorted(unique_test_clients):
                    global_categorizations_sanity[client_id] = global_pred_categories
                
                consistency_metrics_global = compute_consistency_metrics(categorizations=global_categorizations_sanity, prefix="consistency_global", risk_names=RISK_NAMES, client_ids=sorted(unique_test_clients))
                
                if consistency_metrics_global:
                    for risk_name in RISK_NAMES:
                        any_key = f"consistency_global/inconsistency_any_risk_{risk_name}"
                        dist_key = f"consistency_global/inconsistency_distance_risk_{risk_name}"
                        disagree_key = f"consistency_global/patient_disagreement_risk_{risk_name}"
                        
                        if any_key in consistency_metrics_global:
                            print(f"\n{risk_name}:")
                            print(f"Inconsistency (any): {consistency_metrics_global[any_key]:.6f} (expected: 0.0)")
                            print(f"Inconsistency (distance): {consistency_metrics_global[dist_key]:.6f} (expected: 0.0)")
                            print(f"Patient disagreement: {consistency_metrics_global[disagree_key]:.6f} (expected: 0.0)")
                    
                    if "consistency_global/inconsistency_any_macro" in consistency_metrics_global:
                        print(f"\nMacro Averages (Global Thresholds):")
                        print(f"Inconsistency (any): {consistency_metrics_global['consistency_global/inconsistency_any_macro']:.6f} (expected: 0.0)")
                        print(f"Inconsistency (distance): {consistency_metrics_global['consistency_global/inconsistency_distance_macro']:.6f} (expected: 0.0)")
                        print(f"Patient disagreement: {consistency_metrics_global['consistency_global/patient_disagreement_macro']:.6f} (expected: 0.0)")
                    
                    all_test_metrics.update(consistency_metrics_global)
    
    log_metrics_wandb(all_test_metrics, prefix="test/")
    
    print("\nDone")
    wandb.finish()

if __name__ == '__main__':
    main()