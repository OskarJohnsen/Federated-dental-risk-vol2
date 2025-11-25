import wandb
import torch
from torch.optim.lr_scheduler import StepLR
from ..models.base.trainer import BaseTrainer
from ...core.paths import root_path
from ..models.architectures.mlp import MLP
from ..data.loaders import load_data_with_split
from ..data.datasets import create_data_loaders
from ..constants import RISK_NAMES
from ..metrics.calc_metrics import dataset_metrics
from ..metrics.threshold import optimize_thresholds_f1, optimize_thresholds_youden
from ..metrics.report import log_metrics_wandb, log_dataset_info, log_thresholds, log_experiment_config, metrics_summary
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
        tags=[config.experiment_type, config.model, f"seed{config.model_seed}"],
        group=config.get_group_name(),
        job_type=config.experiment_type,
    )
    
    if config.data_version is None:
        data_path = root_path('data', 'raw', 'fed_recommenders_synthetic_dataset.csv')
        config.data_version = get_data_version(str(data_path))
    
    print(f"{config.experiment_type.upper()} TRAINING")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Seed (Model): {config.model_seed}, Seed (Data): {config.data_split_seed}")

    print("\nData")
    data = load_data_with_split(config=config)
    n_features = data['train']['X'].shape[1]
    print(f"Features: {n_features}")
    print(f"Train samples: {len(data['train']['X'])}")
    print(f"Val samples: {len(data['val']['X'])}")
    print(f"Test samples: {len(data['test']['X'])}")
    
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

    for epoch, (train_loss, val_loss) in enumerate(zip(
        results['history']['train_loss'],
        results['history']['val_loss']
    ), 1):
        epoch_metrics = {
            "train/loss": train_loss,
            "val/loss": val_loss,
        }
        log_metrics_wandb(epoch_metrics, prefix="", epoch=epoch)

    if config.threshold_method == "youden":
        print("\nTHRESHOLD TUNING (Youdens J)")
        val_metrics = trainer.evaluate(val_loader)
        val_probs = val_metrics.get("_probs")
        val_labels = val_metrics.get("_labels")
        risk_names = val_metrics.get("_risk_names", RISK_NAMES)
        optimized_thresholds = optimize_thresholds_youden(val_probs, val_labels, risk_names)
        log_thresholds(optimized_thresholds, val_probs, val_labels, risk_names, method="youden")
    elif config.threshold_method == "f1":
        print("\nTHRESHOLD TUNING (F1)")
        val_metrics = trainer.evaluate(val_loader)
        val_probs = val_metrics.get("_probs")
        val_labels = val_metrics.get("_labels")
        risk_names = val_metrics.get("_risk_names", RISK_NAMES)
        optimized_thresholds = optimize_thresholds_f1(val_probs, val_labels, risk_names)
        log_thresholds(optimized_thresholds, val_probs, val_labels, risk_names, method="f1")
    
    final_metrics = trainer.evaluate(test_loader, thresholds=optimized_thresholds)
    
    log_metrics_wandb(final_metrics, prefix="test/")
    metrics_summary(final_metrics)
    
    print("\nDone")
    wandb.finish()

if __name__ == '__main__':
    main()