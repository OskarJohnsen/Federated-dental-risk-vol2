import wandb
import torch
from torch.optim.lr_scheduler import StepLR
from ..models.base.trainer import BaseTrainer
from ...core.paths import root_path
from ..models.architectures.mlp import MLP
from ..data.loaders import load_data_per_client, load_raw_data, load_global_test_set
from ..data.datasets import create_data_loaders
from ..constants import RISK_NAMES
from ..metrics.calc_metrics import dataset_metrics, model_metrics_categories, compute_consistency_metrics
from ..metrics.threshold import percentile_thresholds, apply_risk_categorization, load_global_thresholds
from ..metrics.report import log_metrics_wandb, log_dataset_info, log_experiment_config, log_partition_metadata
from ..util.seed import all_seeds
from ..util.wandb_config import get_wandb_project, get_wandb_entity
from ..config.experiment_config import ExperimentConfig, get_data_version
from typing import Optional
import numpy as np
from ..constants import DATASET, IID_TYPE

def main(config: ExperimentConfig):
    all_seeds(config.model_seed)

    wandb.init(
        project=get_wandb_project(),
        entity=get_wandb_entity(),
        name=config.get_run_name(),
        config=config.to_wandb_config(),
        tags=[config.experiment_type, config.model, f"seed{config.model_seed}", "percentile"],
        group=config.get_group_name(),
        job_type=config.experiment_type,
    )

    if config.dataset_path is None:
        config.dataset_path = str(root_path('data', 'raw', f'synthetic_dataset_{DATASET}_{IID_TYPE}.csv'))
    if config.test_set_path is None:
        config.test_set_path = str(root_path('data', 'processed', 'global_test_set.csv'))
    
    if config.data_version is None:
        config.data_version = get_data_version(config.dataset_path)
    
    print(f"{config.experiment_type.upper()} TRAINING")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Seed (Model): {config.model_seed}, Seed (Data): {config.data_split_seed}")
    
    print("\nLoading full dataset...")
    full_data = load_raw_data(dataset_path=config.dataset_path)
    client_ids = sorted(full_data['Client'].unique())
    print(f"Found {len(client_ids)} clients: {client_ids}")
    
    global_thresholds = load_global_thresholds(root_path("configs", "global_thresholds", f'{DATASET}', f'global_thresholds_{IID_TYPE}.json'))
    
    log_partition_metadata(global_thresholds)
    
    all_client_metrics = {}
    all_client_thresholds = {}
    all_client_categorizations_per_client = {}
    all_client_categorizations_global = {}
    all_client_category_metrics = {}
    
    for client_id in client_ids:
        print(f"\nTraining Client {client_id}")
        
        client_data = load_data_per_client(full_data, client_id, config=config)
        
        n_features = client_data['train']['X'].shape[1]
        print(f"Features: {n_features}")
        print(f"Train samples: {len(client_data['train']['X']):,}")
        print(f"Val samples: {len(client_data['val']['X']):,}")
        
        y_train = client_data["train"]["y_classification"]
        y_val = client_data["val"]["y_classification"]
        
        preprocessing_pipeline = client_data.get('_preprocessing_pipeline')
        if preprocessing_pipeline is None:
            raise ValueError(f"Preprocessing pipeline not found for client {client_id}")

        client_global_test_data = load_global_test_set(test_set_path=config.test_set_path, preprocessing_pipeline=preprocessing_pipeline)
        
        global_test_X = client_global_test_data['X'].copy()
        
        missing_cols = set(client_data['train']['X'].columns) - set(global_test_X.columns)
        extra_cols = set(global_test_X.columns) - set(client_data['train']['X'].columns)
        
        if missing_cols:
            print(f"STOP: Test set missing {len(missing_cols)} columns from training.")
        if extra_cols:
            print(f"STOP: Test set has {len(extra_cols)} extra columns.")
        
        global_test_X = global_test_X.reindex(columns=client_data['train']['X'].columns, fill_value=0.0)
        
        client_data['test'] = {
            'X': global_test_X,
            'y_classification': client_global_test_data['y_classification']
        }
        if 'y_probabilities' in client_global_test_data:
            client_data['test']['y_probabilities'] = client_global_test_data['y_probabilities']
        if 'y_categories' in client_global_test_data:
            client_data['test']['y_categories'] = client_global_test_data['y_categories']
        
        y_test = client_data["test"]["y_classification"]
        
        dataset_metrics_log = dataset_metrics(client_data, y_train, y_test, RISK_NAMES)
        log_dataset_info(dataset_metrics_log, RISK_NAMES)
        
        train_loader, val_loader, test_loader = create_data_loaders(client_data, batch_size=config.batch_size)
        
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
            pos_weights = torch.clamp(
                torch.tensor(neg_counts / (pos_counts + eps), dtype=torch.float32), 
                min=1.0, 
                max=max_weight
            )
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
                f"client_{client_id}/train/loss": train_loss,
                f"client_{client_id}/val/loss": val_loss,
            }
            log_metrics_wandb(epoch_metrics, prefix="", epoch=epoch)
        
        val_metrics = trainer.evaluate(val_loader)
        val_probs = val_metrics.get("_probs")
        val_labels = val_metrics.get("_labels")

        categorization_boundaries = percentile_thresholds(val_probs=val_probs, percentiles=[33, 67], risk_names=RISK_NAMES)
        
        print(f"Categorization boundaries for Client {client_id}:")
        for risk_name, boundaries in categorization_boundaries.items():
            low_med = boundaries.get("low_medium_boundary", 0.33)
            med_high = boundaries.get("medium_high_boundary", 0.67)
            print(f"{risk_name}:")
            print(f"Low: < {low_med}")
            print(f"Medium: {low_med} - {med_high}")
            print(f"High: >= {med_high}")
        
        final_metrics = trainer.evaluate(test_loader, thresholds=None)
        
        test_probs = final_metrics.get("_probs")
        test_true_categories = final_metrics.get("_true_categories")
        
        client_test_metrics = {}
        
        prob_keys = [k for k in final_metrics.keys() if k.startswith(('mse_', 'mae_', 'ece_prob_'))]
        for key in prob_keys:
            client_test_metrics[f"client_{client_id}/test/{key}"] = final_metrics[key]
        
        if config.category_strategy in ["per_client", "both"] and test_probs is not None:
            per_client_pred_categories = apply_risk_categorization(test_probs, categorization_boundaries, risk_names=RISK_NAMES)
            all_client_categorizations_per_client[client_id] = per_client_pred_categories.astype(np.uint8)
            
            if test_true_categories is not None:
                per_client_cat_metrics = model_metrics_categories(per_client_pred_categories, test_true_categories, risk_names=RISK_NAMES, prefix=f"client_{client_id}/test/category_per_client")
                client_test_metrics.update(per_client_cat_metrics)
        
        if config.category_strategy in ["global", "both"] and test_probs is not None:
            global_pred_categories = apply_risk_categorization(test_probs, global_thresholds, risk_names=RISK_NAMES)
            all_client_categorizations_global[client_id] = global_pred_categories.astype(np.uint8)
            
            if test_true_categories is not None:
                global_cat_metrics = model_metrics_categories(global_pred_categories, test_true_categories,risk_names=RISK_NAMES, prefix=f"client_{client_id}/test/category_global")
                client_test_metrics.update(global_cat_metrics)
        
        all_client_category_metrics[client_id] = {
        k: v for k, v in client_test_metrics.items()
        if "/test/category_" in k
        }

        all_client_metrics[client_id] = final_metrics
        all_client_thresholds[client_id] = categorization_boundaries
        
        log_metrics_wandb(client_test_metrics, prefix="")
        
        print(f"\nClient {client_id} - Probability metrics:")
        for risk_name in RISK_NAMES:
            mse_key = f"mse_risk_{risk_name}"
            mae_key = f"mae_risk_{risk_name}"
            if mse_key in final_metrics:
                print(f"{risk_name}: MSE={final_metrics[mse_key]:.6f}, MAE={final_metrics[mae_key]:.6f}")
        
        del model
        del trainer
        del optimizer
        if scheduler is not None:
            del scheduler
        del train_loader
        del val_loader
        del test_loader
        del client_data
        del client_global_test_data
    
    # Per-client
    if config.category_strategy in ["per_client", "both"] and len(all_client_categorizations_per_client) > 1:
        print("\nCONSISTENCY METRICS: Per-Client Thresholds (Local Models)")
        print("(Different models, different thresholds per client)")
        
        categorizations_for_consistency = {k: v.astype(np.int32) for k, v in all_client_categorizations_per_client.items()}
        consistency_metrics_per_client = compute_consistency_metrics(categorizations=categorizations_for_consistency, prefix="consistency_per_client", risk_names=RISK_NAMES, client_ids=client_ids)
        
        if consistency_metrics_per_client:
            for risk_name in RISK_NAMES:
                disagree_key = f"consistency_per_client/patient_disagreement_risk_{risk_name}"
                
                if disagree_key in consistency_metrics_per_client:
                    print(f"\n{risk_name}:")
                    print(f"Patient disagreement: {consistency_metrics_per_client[disagree_key]:.4f}")
            
            if "consistency_per_client/patient_disagreement_macro" in consistency_metrics_per_client:
                print(f"\nMacro Averages (Per-Client Thresholds):")
                print(f"Patient disagreement: {consistency_metrics_per_client['consistency_per_client/patient_disagreement_macro']:.4f}")
            
            log_metrics_wandb(consistency_metrics_per_client, prefix="")
    elif config.category_strategy in ["per_client", "both"]:
        print("Only one client model: skipping per-client consistency metrics")
    
        # Global
    if config.category_strategy in ["global", "both"] and len(all_client_categorizations_global) > 1:
        print("\nCONSISTENCY METRICS: Global Thresholds (Local Models)")
        print("(Different models, same global thresholds)")
        
        categorizations_for_consistency = {
            k: v.astype(np.int32) for k, v in all_client_categorizations_global.items()
        }
        consistency_metrics_global = compute_consistency_metrics(
            categorizations=categorizations_for_consistency,
            prefix="consistency_global",
            risk_names=RISK_NAMES,
            client_ids=client_ids,
        )
        
        if consistency_metrics_global:
            for risk_name in RISK_NAMES:
                disagree_key = f"consistency_global/patient_disagreement_risk_{risk_name}"
                
                if disagree_key in consistency_metrics_global:
                    print(f"\n{risk_name}:")
                    print(f"Patient disagreement: {consistency_metrics_global[disagree_key]:.4f}")
            
            if "consistency_global/patient_disagreement_macro" in consistency_metrics_global:
                print("\nMacro Averages (Global Thresholds):")
                print(
                    f"Patient disagreement: "
                    f"{consistency_metrics_global['consistency_global/patient_disagreement_macro']:.4f}"
                )
            
            log_metrics_wandb(consistency_metrics_global, prefix="")
    elif config.category_strategy in ["global", "both"]:
        print("Only one client model: skipping global consistency metrics")
    
    # --- Byg summary_metrics til sweepet ------------------------------------
    summary_metrics = {}

    # 1) Probability-metrics: gennemsnit over klienter
    prob_values_per_key: dict[str, list[float]] = {}

    for cid in client_ids:
        cm = all_client_metrics.get(cid, {})
        for key, value in cm.items():
            if key.startswith(("mse_", "mae_", "ece_prob_")) and value is not None:
                prob_values_per_key.setdefault(key, []).append(float(value))

    for key, vals in prob_values_per_key.items():
        if vals:
            summary_metrics[key] = float(np.mean(vals))

    # 2) Category-metrics: gennemsnit over klienter
    category_values_per_key: dict[str, list[float]] = {}

    for cid in client_ids:
        cm = all_client_category_metrics.get(cid, {})
        for key, value in cm.items():
            if value is None:
                continue

            clean_key = key.replace(f"client_{cid}/test/", "")
            category_values_per_key.setdefault(clean_key, []).append(float(value))

    for key, vals in category_values_per_key.items():
        if vals:
            summary_metrics[key] = float(np.mean(vals))

    # 3) Tilføj consistency-metrics
    if "consistency_metrics_per_client" in locals() and consistency_metrics_per_client:
        summary_metrics.update(consistency_metrics_per_client)

    if "consistency_metrics_global" in locals() and consistency_metrics_global:
        summary_metrics.update(consistency_metrics_global)

    # 4) Lav færdige macro-metrics på tværs af komplikationer
    def add_macro_summary(input_prefix: str, output_key: str):
        vals = []
        for risk in RISK_NAMES:
            key = f"{input_prefix}{risk}"
            if key in summary_metrics and summary_metrics[key] is not None:
                vals.append(summary_metrics[key])
        if vals:
            summary_metrics[output_key] = float(np.mean(vals))

    add_macro_summary("category_global_f1_macro_risk_", "f1_global_macro")
    add_macro_summary("category_per_client_f1_macro_risk_", "f1_per_client_macro")
    add_macro_summary("mse_risk_", "mse_macro")
    add_macro_summary("ece_prob_risk_", "ece_macro")


    wandb.finish()
    return summary_metrics

if __name__ == '__main__':
    main()

