import wandb
import torch
from torch.optim.lr_scheduler import StepLR
from ..models.base.trainer import BaseTrainer
from ..models.architectures.mlp import MLP
from ..data.loaders import load_data_with_split
from ..data.datasets import create_data_loaders
from ..constants import RISK_NAMES
from ..metrics.calc_metrics import dataset_metrics
from ..metrics.threshold import optimize_thresholds_f1, optimize_thresholds_youden
from ..metrics.report import log_metrics_wandb, print_dataset_metrics, print_thresholds,log_thresholds_wandb, metrics_summary
import numpy as np

def main():
    wandb.init(
        project="digit-federated-recommenders",
        name="MLP_BCELogits_Unweighted_threshold(youden)",
        config={
            "model": "MLP",
            "hidden_size": [128, 64],
            "dropout": 0.2,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "epochs": 15,
            "seed": 42,
        }
    )

    print("CENTRALIZED TRAINING")

    print("\nData")
    data = load_data_with_split()
    n_features = data['train']['X'].shape[1]
    print(f"Features: {n_features}")
    print(f"Train samples: {len(data['train']['X'])}")
    print(f"Val samples: {len(data['val']['X'])}")
    print(f"Test samples: {len(data['test']['X'])}")
    
    y_train = data["train"]["y_classification"]
    y_val = data["val"]["y_classification"]
    y_test = data["test"]["y_classification"]
    
    dataset_metrics_log = dataset_metrics(data, y_train, y_test, RISK_NAMES)
    print_dataset_metrics(dataset_metrics_log, RISK_NAMES)
    wandb.log(dataset_metrics_log)
    
    train_loader, val_loader, test_loader = create_data_loaders(data, batch_size=wandb.config.batch_size)
    
    model = MLP(
        input_size=n_features,
        hidden_size=wandb.config.hidden_size,
        dropout=wandb.config.dropout,
        n_clf_classes=4,
    )
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_counts = y_train.sum(axis=0).values.astype(float)
    total = len(y_train)
    neg_counts = total - pos_counts
    eps = 1e-6
    max_weight = 100.0
    pos_weights = torch.clamp(torch.tensor(neg_counts / (pos_counts + eps), dtype=torch.float32), min=1.0, max=max_weight)
    loss_clf = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        loss_clf=loss_clf,
        scheduler=scheduler,
        experiment_type='centralized',
        seed=wandb.config.seed
    )
    
    print(f"Training on device: {trainer.device}")
    wandb.config.update({
        "model/total_parameters": sum(p.numel() for p in model.parameters()),
    })
    
    results = trainer.fit(train_loader, val_loader=val_loader, epochs=wandb.config.epochs)

    for epoch, (train_loss, val_loss) in enumerate(zip(
        results['history']['train_loss'],
        results['history']['val_loss']
    ), 1):
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
        })

    threshold_method = "youden"  # "youden" or "f1"
    
    if threshold_method == "youden":
        print("\nTHRESHOLD TUNING (Youdens J)")
        val_metrics = trainer.evaluate(val_loader)
        val_probs = val_metrics.get("_probs")
        val_labels = val_metrics.get("_labels")
        risk_names = val_metrics.get("_risk_names", RISK_NAMES)
        optimized_thresholds = optimize_thresholds_youden(val_probs, val_labels, risk_names)
        print_thresholds(optimized_thresholds, val_probs, val_labels, risk_names, method="youden")
    else:
        print("\nTHRESHOLD TUNING (F1)")
        val_metrics = trainer.evaluate(val_loader)
        val_probs = val_metrics.get("_probs")
        val_labels = val_metrics.get("_labels")
        risk_names = val_metrics.get("_risk_names", RISK_NAMES)
        optimized_thresholds = optimize_thresholds_f1(val_probs, val_labels, risk_names)
        print_thresholds(optimized_thresholds, val_probs, val_labels, risk_names, method="f1")
    
    log_thresholds_wandb(optimized_thresholds)
    
    final_metrics = trainer.evaluate(test_loader, thresholds=optimized_thresholds)
    
    log_metrics_wandb(final_metrics, prefix="test/")
    metrics_summary(final_metrics)
    
    print("\nDone")
    wandb.finish()

if __name__ == '__main__':
    main()