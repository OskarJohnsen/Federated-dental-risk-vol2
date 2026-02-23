from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fdrp.core.paths import root_path
from fdrp.ml.config.experiment_config import ExperimentConfig, get_data_version
from fdrp.ml.constants import DATASET, IID_TYPE
from fdrp.ml.data.loaders import (
    load_data_with_split,
    load_global_test_set,
    load_raw_data,
)
from fdrp.ml.data.datasets import create_data_loaders, MultiTaskDataset
from fdrp.ml.data.preprocessing import PreprocessingPipeline
from fdrp.ml.models.architectures.mlp import MLP
from fdrp.ml.models.base.trainer import BaseTrainer
from fdrp.ml.util.seed import all_seeds


ExperimentType = Literal["centralized", "federated"]


def _evaluate_centralized(
    checkpoint: Path,
    batch_size: int | None = None,
) -> dict:
    """
    Evaluer et CENTRALIZED checkpoint på global test set og returnér test_metrics.
    """
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

    all_seeds(config.model_seed)

    # Brug din eksisterende helper til at loade train/val + preprocessing
    data = load_data_with_split(config=config)
    train_X = data["train"]["X"]
    n_features = train_X.shape[1]

    preprocessing_pipeline = data.get("_preprocessing_pipeline")
    if preprocessing_pipeline is None:
        raise ValueError("Preprocessing pipeline not found in training data.")

    # Load global test set med samme pipeline
    global_test_data = load_global_test_set(
        test_set_path=config.test_set_path,
        preprocessing_pipeline=preprocessing_pipeline,
    )

    global_test_X = global_test_data["X"]
    # Align kolonner med train
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

    # Test-dataloader
    _, _, test_loader = create_data_loaders(
        data,
        batch_size=config.batch_size,
    )

    # Model & trainer (samme arkitektur som træning)
    model = MLP(
        input_size=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        n_clf_classes=4,
    )

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

    checkpoint = checkpoint.expanduser().resolve()
    trainer.load_checkpoint(str(checkpoint))

    test_metrics = trainer.evaluate(test_loader, thresholds=None)
    return test_metrics


def _evaluate_federated(
    checkpoint: Path,
    batch_size: int | None = None,
) -> dict:
    """
    Evaluer et FEDERATED checkpoint på global test set og returnér test_metrics.
    Genskaber global preprocessing pipeline som i federated-train.
    """
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

    all_seeds(config.model_seed)

    # --------- byg global preprocessing pipeline som i federated.train ---------
    full_data = load_raw_data(dataset_path=config.dataset_path)
    client_ids = sorted(full_data["Client"].unique())

    all_training_data = []
    for client_id in client_ids:
        client_mask = full_data["Client"] == client_id
        X_client = full_data["X"][client_mask].copy()
        val_size_adjusted = config.val_size / (1 - config.test_size)
        X_train, _, _, _ = train_test_split(
            X_client,
            full_data["y_classification"][client_mask],
            test_size=val_size_adjusted,
            random_state=config.data_split_seed + client_id,
        )
        all_training_data.append(X_train)

    combined_training_data = pd.concat(all_training_data, ignore_index=True)
    global_preprocessing_pipeline = PreprocessingPipeline()
    global_preprocessing_pipeline.fit(combined_training_data)

    n_features = len(global_preprocessing_pipeline.feature_columns)

    # --------- global test set med denne pipeline ---------
    global_test_data = load_global_test_set(
        test_set_path=config.test_set_path,
        preprocessing_pipeline=global_preprocessing_pipeline,
    )

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

    from torch.utils.data import DataLoader

    test_size = len(test_dataset)
    test_batch_size = min(config.batch_size, test_size) if test_size > 0 else config.batch_size
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

    checkpoint = checkpoint.expanduser().resolve()
    trainer.load_checkpoint(str(checkpoint))

    test_metrics = trainer.evaluate(test_loader, thresholds=None)
    return test_metrics


def evaluate_checkpoint(
    experiment_type: ExperimentType,
    checkpoint: Path,
    batch_size: int | None = None,
) -> dict:
    """
    Fælles entrypoint du kan importere i notebooks/scripts.

    experiment_type: "centralized" eller "federated"

    Returnerer det dict, som BaseTrainer.evaluate() giver:
      - scalar metrics (mse_*, mae_*, ece_*, loss_clf, ...)
      - "_probs":           (n_samples, n_risks)
      - "_true_probs":      (n_samples, n_risks)
      - "_true_categories": (n_samples, n_risks) hvis tilgængelig
    """
    et = experiment_type.lower()
    if et not in {"centralized", "federated"}:
        raise ValueError("experiment_type skal være 'centralized' eller 'federated'")

    if et == "centralized":
        return _evaluate_centralized(checkpoint, batch_size=batch_size)
    else:
        return _evaluate_federated(checkpoint, batch_size=batch_size)
    

"""
Sådan bruges koden i et andet document:
from pathlib import Path
from fdrp.analysis.eval_utils import evaluate_checkpoint

# Centralized-model
ckpt_central = Path("checkpoints/centralized/MLP_20260216_092657.pt")
metrics_central = evaluate_checkpoint("centralized", ckpt_central)

# Federated-model
ckpt_fed = Path("checkpoints/federated/MLP_20260220_123456.pt")
metrics_fed = evaluate_checkpoint("federated", ckpt_fed)
"""