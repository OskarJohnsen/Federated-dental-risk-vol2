from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import typer
import pandas as pd

# Justér disse imports hvis din struktur er lidt anderledes
from fdrp.core.paths import root_path
from fdrp.ml.config.experiment_config import ExperimentConfig, get_data_version
from fdrp.ml.constants import DATASET, IID_TYPE
from fdrp.ml.data.loaders import (
    load_data_with_split,
    load_global_test_set,
    load_raw_data,
    load_data_per_client,   # 👈 NY
)
from fdrp.ml.data.datasets import create_data_loaders, MultiTaskDataset
from fdrp.ml.data.preprocessing import PreprocessingPipeline
from fdrp.ml.models.architectures.mlp import MLP
from fdrp.ml.models.base.trainer import BaseTrainer
from fdrp.ml.util.seed import all_seeds


# -------------------------------------------------------------------
# Fælles helper: sample-size analyse fra metrics
# -------------------------------------------------------------------
def sample_size_analysis_from_metrics(
    test_metrics: dict,
    n_boot: int = 100,
    random_state: int = 42,
    title_suffix: str = "",
) -> None:
    """
    Lav et plot af macro MSE som funktion af test-størrelse ved hjælp af bootstrap.

    Forventninger til test_metrics:
      - "_probs":      shape (n_samples, n_risks)
      - "_true_probs": same shape
    """
    probs = test_metrics.get("_probs")
    true_probs = test_metrics.get("_true_probs")

    if probs is None or true_probs is None:
        print("\n[Sample size] Skipper analyse: _probs eller _true_probs mangler i test_metrics.")
        return

    n_samples, n_risks = probs.shape
    print(f"\n[Sample size] Kører analyse på {n_samples} test-samples ({n_risks} risici).")

    # squared error per sample per risk
    squared_errors = (probs - true_probs) ** 2  # (n_samples, n_risks)

    # Vælg test-størrelser automatisk: 6 punkter mellem min(50, n_samples) og n_samples
    min_n = min(50, n_samples)
    n_points = 10
    test_sizes = np.unique(
        np.linspace(min_n, n_samples, n_points, dtype=int)
    )
    test_sizes = [int(n) for n in test_sizes if n <= n_samples]

    rng = np.random.default_rng(random_state)

    mean_mse_macro = []
    ci95_mse_macro = []

    for n in test_sizes:
        mses = []
        for _ in range(n_boot):
            idx = rng.choice(n_samples, size=n, replace=True)
            mse_per_risk = squared_errors[idx].mean(axis=0)
            mse_macro = mse_per_risk.mean()
            mses.append(mse_macro)

        mses = np.array(mses)
        mean_mse = mses.mean()
        std_mse = mses.std(ddof=1)
        ci95 = 1.96 * std_mse

        mean_mse_macro.append(mean_mse)
        ci95_mse_macro.append(ci95)

        print(f"[n={n:5d}] macro MSE ~ {mean_mse:.6f} ± {ci95:.6f} (95% CI)")

    mean_mse_macro = np.array(mean_mse_macro)
    ci95_mse_macro = np.array(ci95_mse_macro)

    # Plot
    plt.figure()
    plt.errorbar(
        test_sizes,
        mean_mse_macro,
        yerr=ci95_mse_macro,
        marker="o",
        linestyle="-",
        capsize=4,
    )
    plt.xlabel("Test set size (number of patients)")
    plt.ylabel("Macro MSE (mean ± 95% CI)")
    plt.title("Stabilitet af macro MSE som funktion af test-størrelse" + title_suffix)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Centralized analyse
# -------------------------------------------------------------------
def analyze_centralized(checkpoint: Path, batch_size: int | None) -> None:
    dataset_path = root_path("data", "raw", f"synthetic_dataset_{DATASET}_{IID_TYPE}.csv")
    test_set_path = root_path("data", "processed", f"{DATASET}", f"global_test_set_{IID_TYPE}.csv")

    config = ExperimentConfig(
        experiment_type="centralized",
        experiment_id=f"{DATASET}_{IID_TYPE}",
        model_seed=42,
        data_split_seed=42,
        dataset_path=str(dataset_path),
        test_set_path=str(test_set_path),
    )

    if config.data_version is None:
        config.data_version = get_data_version(config.dataset_path)

    if batch_size is not None:
        config.batch_size = batch_size

    print("=== Sample size stability analysis (CENTRALIZED) ===")
    print(f"Dataset path:      {config.dataset_path}")
    print(f"Global test path:  {config.test_set_path}")
    print(f"Checkpoint:        {checkpoint}")
    print(f"Batch size:        {config.batch_size}")

    all_seeds(config.model_seed)

    # Load data + preprocessing pipeline via den eksisterende helper
    data = load_data_with_split(config=config)
    n_features = data["train"]["X"].shape[1]
    print(f"Features (train):  {n_features}")

    preprocessing_pipeline = data.get("_preprocessing_pipeline")
    if preprocessing_pipeline is None:
        raise ValueError("Preprocessing pipeline not found in training data.")

    global_test_data = load_global_test_set(
        test_set_path=config.test_set_path,
        preprocessing_pipeline=preprocessing_pipeline,
    )
    print(f"Global test samples (raw): {len(global_test_data['X']):,}")

    train_X = data["train"]["X"]
    global_test_X = global_test_data["X"]

    # Kolonne-align som i centralized-train
    global_test_X = global_test_X.reindex(columns=train_X.columns, fill_value=0.0)

    data["test"] = {
        "X": global_test_X,
        "y_classification": global_test_data["y_classification"],
    }
    if "y_probabilities" in global_test_data:
        data["test"]["y_probabilities"] = global_test_data["y_probabilities"]
    if "y_categories" in global_test_data:
        data["test"]["y_categories"] = global_test_data["y_categories"]
    if "Client" in global_test_data:
        data["test"]["Client"] = global_test_data["Client"]

    # Dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data,
        batch_size=config.batch_size,
    )

    # Model + trainer (samme arkitektur)
    model = MLP(
        input_size=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        n_clf_classes=4,
    )
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # class weights hvis slået til
    y_train = data["train"]["y_classification"]
    pos_weights = None
    if getattr(config, "use_class_weights", False):
        pos_counts = y_train.sum(axis=0).values.astype(float)
        total = len(y_train)
        neg_counts = total - pos_counts
        eps = 1e-6
        max_weight = 100.0
        pos_weights = torch.clamp(
            torch.tensor(neg_counts / (pos_counts + eps), dtype=torch.float32),
            min=1.0,
            max=max_weight,
        )
        loss_clf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        loss_clf = torch.nn.BCEWithLogitsLoss()

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer in config: {config.optimizer}")

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        loss_clf=loss_clf,
        scheduler=None,
        experiment_type=config.experiment_type,
        seed=config.model_seed,
    )

    print(f"Evaluating on device: {trainer.device}")

    # Load checkpoint
    checkpoint = checkpoint.expanduser().resolve()
    print(f"\nLoading checkpoint: {checkpoint}")
    trainer.load_checkpoint(str(checkpoint))

    # Eval + analyse
    print("\nEVAL GLOBAL TEST SET")
    test_metrics = trainer.evaluate(test_loader, thresholds=None)

    print("\nEvaluation results (scalar metrics):")
    for k, v in test_metrics.items():
        if not k.startswith("_"):
            print(f"{k}: {v}")

    sample_size_analysis_from_metrics(
        test_metrics,
        n_boot=100,
        random_state=42,
        title_suffix=" (centralized)",
    )


# -------------------------------------------------------------------
# Federated analyse
# -------------------------------------------------------------------
def analyze_federated(checkpoint: Path, batch_size: int | None) -> None:
    dataset_path = root_path("data", "raw", f"synthetic_dataset_{DATASET}_{IID_TYPE}.csv")
    test_set_path = root_path("data", "processed", f"{DATASET}", f"global_test_set_{IID_TYPE}.csv")

    config = ExperimentConfig(
        experiment_type="federated",
        experiment_id=f"{DATASET}_{IID_TYPE}",
        model_seed=42,
        data_split_seed=42,
        dataset_path=str(dataset_path),
        test_set_path=str(test_set_path),
        federated_rounds=6,
        local_epochs=5,
    )

    if config.data_version is None:
        config.data_version = get_data_version(config.dataset_path)

    if batch_size is not None:
        config.batch_size = batch_size

    print("=== Sample size stability analysis (FEDERATED) ===")
    print(f"Dataset path:      {config.dataset_path}")
    print(f"Global test path:  {config.test_set_path}")
    print(f"Checkpoint:        {checkpoint}")
    print(f"Batch size:        {config.batch_size}")

    all_seeds(config.model_seed)

    # --------- byg global preprocessing pipeline som i federated.train ---------
    print("\nLoading full dataset...")
    full_data = load_raw_data(dataset_path=config.dataset_path)
    client_ids = sorted(full_data["Client"].unique())
    print(f"Found {len(client_ids)} clients: {client_ids}")

    print("\nCreating global preprocessing pipeline from all training data...")
    all_training_data = []
    for client_id in client_ids:
        client_mask = full_data["Client"] == client_id
        X_client = full_data["X"][client_mask].copy()
        val_size_adjusted = config.val_size / (1 - config.test_size)
        X_train, _, _, _ = (
            # vi bruger samme split-ide som i federated.train
            __import__("sklearn").model_selection.train_test_split(
                X_client,
                full_data["y_classification"][client_mask],
                test_size=val_size_adjusted,
                random_state=config.data_split_seed + client_id,
            )
        )
        all_training_data.append(X_train)

    combined_training_data = pd.concat(all_training_data, ignore_index=True)
    global_preprocessing_pipeline = PreprocessingPipeline()
    global_preprocessing_pipeline.fit(combined_training_data)
    print(f"Global preprocessing pipeline fitted on {len(combined_training_data):,} samples")

    n_features = len(global_preprocessing_pipeline.feature_columns)
    print(f"Features: {n_features}")

    # --------- global test set med denne pipeline ---------
    global_test_data = load_global_test_set(
        test_set_path=config.test_set_path,
        preprocessing_pipeline=global_preprocessing_pipeline,
    )
    print(f"Global test samples: {len(global_test_data['X']):,}")

    global_test_X = global_test_data["X"].copy()
    global_test_X = global_test_X.reindex(
        columns=global_preprocessing_pipeline.feature_columns,
        fill_value=0.0,
    )

    test_data = {
        "X": global_test_X,
        "y_classification": global_test_data["y_classification"],
    }
    if "y_probabilities" in global_test_data:
        test_data["y_probabilities"] = global_test_data["y_probabilities"]
    if "y_categories" in global_test_data:
        test_data["y_categories"] = global_test_data["y_categories"]
    if "Client" in global_test_data:
        test_data["Client"] = global_test_data["Client"]

    y_test_probs = test_data.get("y_probabilities", None)
    y_test_categories = test_data.get("y_categories", None)
    test_dataset = MultiTaskDataset(
        test_data["X"],
        test_data["y_classification"],
        y_probabilities=y_test_probs,
        y_categories=y_test_categories,
    )
    test_size = len(test_dataset)
    test_batch_size = min(config.batch_size, test_size) if test_size > 0 else config.batch_size

    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    # --------- model + trainer ---------
    global_model = MLP(
        input_size=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        n_clf_classes=4,
    )
    print(f"\nGlobal Model: {global_model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    # loss + optimizer (bruges kun til at bygge trainer)
    y_full = full_data["y_classification"]
    pos_weights = None
    if getattr(config, "use_class_weights", False):
        pos_counts = y_full.sum(axis=0).values.astype(float)
        total = len(y_full)
        neg_counts = total - pos_counts
        eps = 1e-6
        max_weight = 100.0
        pos_weights = torch.clamp(
            torch.tensor(neg_counts / (pos_counts + eps), dtype=torch.float32),
            min=1.0,
            max=max_weight,
        )
        loss_clf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        loss_clf = torch.nn.BCEWithLogitsLoss()

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            global_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            global_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer in config: {config.optimizer}")

    trainer = BaseTrainer(
        model=global_model,
        optimizer=optimizer,
        loss_clf=loss_clf,
        scheduler=None,
        experiment_type=config.experiment_type,
        seed=config.model_seed,
    )

    print(f"Evaluating on device: {trainer.device}")

    # Load checkpoint (den trænte federated-model)
    checkpoint = checkpoint.expanduser().resolve()
    print(f"\nLoading checkpoint: {checkpoint}")
    trainer.load_checkpoint(str(checkpoint))

    # Eval + analyse
    print("\nEVAL GLOBAL TEST SET (federated)")
    test_metrics = trainer.evaluate(test_loader, thresholds=None)

    print("\nEvaluation results (scalar metrics):")
    for k, v in test_metrics.items():
        if not k.startswith("_"):
            print(f"{k}: {v}")

    sample_size_analysis_from_metrics(
        test_metrics,
        n_boot=100,
        random_state=42,
        title_suffix=" (federated)",
    )


# -------------------------------------------------------------------
# Local analyse (per klient)
# Er lidt itvivl om dette er den rigtige måde til local.
# -------------------------------------------------------------------
def analyze_local(checkpoint: Path, batch_size: int | None, client_id: int) -> None:
    dataset_path = root_path("data", "raw", f"synthetic_dataset_{DATASET}_{IID_TYPE}.csv")
    test_set_path = root_path("data", "processed", f"{DATASET}", f"global_test_set_{IID_TYPE}.csv")

    config = ExperimentConfig(
        experiment_type="local",
        experiment_id=f"{DATASET}_{IID_TYPE}",
        model_seed=42,
        data_split_seed=42,
        dataset_path=str(dataset_path),
        test_set_path=str(test_set_path),
    )

    if config.data_version is None:
        config.data_version = get_data_version(config.dataset_path)

    if batch_size is not None:
        config.batch_size = batch_size

    print("=== Sample size stability analysis (LOCAL) ===")
    print(f"Dataset path:      {config.dataset_path}")
    print(f"Global test path:  {config.test_set_path}")
    print(f"Checkpoint:        {checkpoint}")
    print(f"Batch size:        {config.batch_size}")
    print(f"Client ID:         {client_id}")

    all_seeds(config.model_seed)

    # ---- load fuldt datasæt og this client's data ----
    print("\nLoading full dataset...")
    full_data = load_raw_data(dataset_path=config.dataset_path)
    client_ids = sorted(full_data["Client"].unique())
    print(f"Found {len(client_ids)} clients: {client_ids}")
    if client_id not in client_ids:
        raise ValueError(f"Client {client_id} not found in dataset (available: {client_ids})")

    print(f"\nLoading data for client {client_id}...")
    client_data = load_data_per_client(full_data, client_id, config=config)

    n_features = client_data["train"]["X"].shape[1]
    print(f"Features (client {client_id}): {n_features}")
    print(f"Train samples: {len(client_data['train']['X']):,}")
    print(f"Val samples:   {len(client_data['val']['X']):,}")

    preprocessing_pipeline = client_data.get("_preprocessing_pipeline")
    if preprocessing_pipeline is None:
        raise ValueError(f"Preprocessing pipeline not found for client {client_id}")

    # ---- global test set gennem denne klients pipeline ----
    client_global_test_data = load_global_test_set(
        test_set_path=config.test_set_path,
        preprocessing_pipeline=preprocessing_pipeline,
    )
    print(f"Global test samples (raw): {len(client_global_test_data['X']):,}")

    global_test_X = client_global_test_data["X"].copy()

    # align kolonner mellem klientens train og global test
    global_test_X = global_test_X.reindex(
        columns=client_data["train"]["X"].columns,
        fill_value=0.0,
    )

    client_data["test"] = {
        "X": global_test_X,
        "y_classification": client_global_test_data["y_classification"],
    }
    if "y_probabilities" in client_global_test_data:
        client_data["test"]["y_probabilities"] = client_global_test_data["y_probabilities"]
    if "y_categories" in client_global_test_data:
        client_data["test"]["y_categories"] = client_global_test_data["y_categories"]

    # ---- dataloaders ----
    train_loader, val_loader, test_loader = create_data_loaders(
        client_data,
        batch_size=config.batch_size,
    )

    # ---- model + trainer (samme arkitektur som local-train) ----
    model = MLP(
        input_size=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        n_clf_classes=4,
    )
    print(f"\nLocal model for client {client_id}: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    y_train = client_data["train"]["y_classification"]
    pos_weights = None
    if getattr(config, "use_class_weights", False):
        pos_counts = y_train.sum(axis=0).values.astype(float)
        total = len(y_train)
        neg_counts = total - pos_counts
        eps = 1e-6
        max_weight = 100.0
        pos_weights = torch.clamp(
            torch.tensor(neg_counts / (pos_counts + eps), dtype=torch.float32),
            min=1.0,
            max=max_weight,
        )
        loss_clf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        loss_clf = torch.nn.BCEWithLogitsLoss()

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer in config: {config.optimizer}")

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        loss_clf=loss_clf,
        scheduler=None,
        experiment_type=config.experiment_type,
        seed=config.model_seed,
    )

    print(f"Evaluating on device: {trainer.device}")

    # ---- load klientens checkpoint ----
    checkpoint = checkpoint.expanduser().resolve()
    print(f"\nLoading checkpoint for client {client_id}: {checkpoint}")
    trainer.load_checkpoint(str(checkpoint))

    # ---- eval + sample-size-analyse ----
    print(f"\nEVAL GLOBAL TEST SET (local model, client {client_id})")
    test_metrics = trainer.evaluate(test_loader, thresholds=None)

    print("\nEvaluation results (scalar metrics):")
    for k, v in test_metrics.items():
        if not k.startswith("_"):
            print(f"{k}: {v}")

    sample_size_analysis_from_metrics(
        test_metrics,
        n_boot=100,
        random_state=42,
        title_suffix=f" (local, client {client_id})",
    )

# -------------------------------------------------------------------
# Typer CLI
# -------------------------------------------------------------------
app = typer.Typer(add_completion=False)


@app.command()
def main(
    experiment_type: str = typer.Argument(..., help="centralized, federated eller local"),
    checkpoint: Path = typer.Argument(..., help="Path til trænet checkpoint (.pt)"),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Override batch size (default: brug værdien fra ExperimentConfig)",
    ),
    client_id: int = typer.Option(
        1,
        "--client-id",
        "-c",
        help="Client ID for local analyse (ignoreres for centralized/federated)",
    ),
):
   
    experiment_type = experiment_type.lower()
    if experiment_type not in {"centralized", "federated", "local"}:
        raise typer.BadParameter("experiment_type skal være 'centralized', 'federated' eller 'local'")

    if experiment_type == "centralized":
        analyze_centralized(checkpoint, batch_size)
    elif experiment_type == "federated":
        analyze_federated(checkpoint, batch_size)
    else:  # local
        analyze_local(checkpoint, batch_size, client_id)

def run():
    app()


if __name__ == "__main__":
    run()


"""
Consol commands:
python -m fdrp.analysis.test_size_stability centralized PATH/TO/CKPT.pt
python -m fdrp.analysis.test_size_stability federated   PATH/TO/CKPT.pt
python -m fdrp.analysis.test_size_stability local path/to/client1.pt --client-id 1
"""