from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import copy
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdrp.ml.config.experiment_config import ExperimentConfig
from fdrp.ml.util.seed import all_seeds
from fdrp.core.paths import root_path, ensure_dir
from fdrp.ml.constants import DATASET, IID_TYPE, RISK_NAMES

from fdrp.data_generation.config.loader import load_all_configs
from fdrp.data_generation.generation.synth import generate_dataset
from fdrp.data_generation.splits import split_global_test_from_pool
from fdrp.data_generation.partitioning.pool_partitioning import (
    partition_dataset_constrained_dirichlet,
    print_partition_statistics,
    compute_partition_heterogeneity_metrics,
    print_quantity_skew_statistics,
    compute_quantity_skew_metrics,
)

from fdrp.ml.federated.train import main as run_federated

"""
This code generates a dataset with the seeds provided in "DATA_SEEDS". I trains the federated model and logs the results to a excel file.
This code was made with ChatGPT.
"""

SUMMARY_PATH = Path(r"C:\Users\oskar\OneDrive\Desktop\Seed_test\federated_seed_sensitivity_summary_four.csv")
PLOT_PATH = Path(r"C:\Users\oskar\OneDrive\Desktop\Seed_test\federated_seed_sensitivity_plot_four.png")

DATA_SEEDS = [1, 2, 3, 4, 5, 6]

MODEL_SEED = 42
DATA_SPLIT_SEED = 42
TEST_SEED = 999

TEST_SIZE = 3000
MIN_SIZE = 100

BETA_L = 1.0
BETA_Q = 1.0


def generate_data_for_seed(seed: int) -> tuple[Path, Path, Dict[str, Any]]:
    print(f"\n[DATA] Generating dataset for seed={seed}")

    configs = load_all_configs(force_reload=True)
    configs["iid_type"] = IID_TYPE

    gen_cfg = configs["generation"]

    if "client_profiles" in gen_cfg:
        gen_cfg["client_profiles"]["seed"] = seed

    np.random.seed(seed)
    gen_cfg["dataset"]["random_seed"] = seed

    pool_cfg = gen_cfg.setdefault("pool_partitioning", {})
    pool_multiplier = int(pool_cfg.get("pool_multiplier", 1))
    pool_cfg["pool_multiplier"] = pool_multiplier
    pool_cfg["test_size"] = int(TEST_SIZE)
    pool_cfg["label_column"] = "Risk_Category_Composite"
    pool_cfg["beta_L"] = float(BETA_L)
    pool_cfg["beta_Q"] = float(BETA_Q)
    pool_cfg["min_size"] = int(MIN_SIZE)

    df_pool, global_thresholds = generate_dataset(configs)

    df_remaining_pool, df_test = split_global_test_from_pool(
        df=df_pool,
        n_test_samples=TEST_SIZE,
        seed=TEST_SEED,
    )

    n_clients = gen_cfg["dataset"]["n_clients"]

    df_train = partition_dataset_constrained_dirichlet(
        df=df_remaining_pool,
        n_clients=n_clients,
        beta_L=BETA_L,
        beta_Q=BETA_Q,
        label_column="Risk_Category_Composite",
        client_column="Client",
        min_size=MIN_SIZE,
        seed=seed,
    )

    print_partition_statistics(df_train, "Risk_Category_Composite", "Client")

    heterogeneity_metrics = compute_partition_heterogeneity_metrics(
        df_train, "Risk_Category_Composite", "Client"
    )

    print_quantity_skew_statistics(df_train, "Client")

    quantity_metrics = compute_quantity_skew_metrics(df_train, "Client")

    partition_metadata = {
        "data_seed": seed,
        "beta_L": BETA_L,
        "beta_Q": BETA_Q,
        "label_column": "Risk_Category_Composite",
        "iid_type": IID_TYPE,
        "partition_method": "constrained_dirichlet",
        "heterogeneity_metrics": heterogeneity_metrics,
        "quantity_skew_metrics": quantity_metrics,
        "final_train_size": int(len(df_train)),
        "test_size": TEST_SIZE,
        "pool_rows": int(len(df_pool)),
        "remaining_rows_after_test_split": int(len(df_remaining_pool)),
        "min_size": MIN_SIZE,
        "pool_multiplier": pool_multiplier,
    }

    global_thresholds["_partition_metadata"] = partition_metadata

    cfg_out_dir = gen_cfg["output"]["output_dir"]
    combo = f"federated_seed_{seed}"
    base = f"synthetic_dataset_{DATASET}_{IID_TYPE}_{combo}"
    proj_root = root_path()

    p = Path(cfg_out_dir)
    if p.is_absolute():
        out_dir = p.resolve()
    else:
        out_dir = proj_root.joinpath(p).resolve()

    try:
        inside_repo = proj_root == out_dir or proj_root in out_dir.parents
    except Exception:
        inside_repo = False

    if not inside_repo:
        out_dir = proj_root.joinpath("data", "raw").resolve()

    ensure_dir(out_dir)

    dataset_csv_path = out_dir.joinpath(f"{base}.csv")
    df_train.to_csv(dataset_csv_path, index=False)

    thresholds_path = (
        proj_root
        / "configs"
        / "global_thresholds"
        / f"{DATASET}"
        / f"global_thresholds_{IID_TYPE}_{combo}.json"
    )
    ensure_dir(thresholds_path.parent)

    with thresholds_path.open("w") as f:
        json.dump(global_thresholds, f, indent=2)

    test_output_path = (
        proj_root
        / "data"
        / "processed"
        / f"{DATASET}"
        / f"global_test_set_{IID_TYPE}_{combo}.csv"
    )
    ensure_dir(test_output_path.parent)

    df_test.to_csv(test_output_path, index=False)

    print(f"[DATA] Saved train dataset to: {dataset_csv_path}")
    print(f"[DATA] Saved global test set to: {test_output_path}")
    print(f"[DATA] Saved thresholds to: {thresholds_path}")

    return dataset_csv_path, test_output_path, partition_metadata


def load_base_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_type="federated",
        experiment_id="seed_sensitivity_federated",
    )


def config_for_run(
    base_config: ExperimentConfig,
    data_seed: int,
    dataset_path: Path,
    testset_path: Path,
) -> ExperimentConfig:

    cfg = copy.deepcopy(base_config)

    cfg.experiment_type = "federated"
    cfg.experiment_id = f"seed_sensitivity_federated_seed_{data_seed}"

    cfg.dataset_path = str(dataset_path)
    cfg.test_set_path = str(testset_path)

    cfg.model_seed = MODEL_SEED
    cfg.data_split_seed = DATA_SPLIT_SEED

    cfg.category_strategy = "both"
    cfg.threshold_method = "percentile"
    cfg.use_wandb = False

    cfg.beta_L = float(BETA_L)
    cfg.beta_Q = float(BETA_Q)
    cfg.run_suffix = f"federated_seed_{data_seed}"

    # Federated settings
    cfg.federated_rounds = 6
    cfg.local_epochs = 5
    cfg.aggregation_method = "fedavg"

    # Hardcode architecture here if needed
    # Example:
    # cfg.hidden_size = [128, 64]
    # cfg.head_hidden_sizes = [128, 64]

    return cfg


def extract_summary_row(
    data_seed: int,
    metrics: Dict[str, Any],
    partition_metadata: Dict[str, Any],
) -> Dict[str, Any]:

    row: Dict[str, Any] = {
        "paradigm": "federated",
        "data_seed": data_seed,
        "model_seed": MODEL_SEED,
        "data_split_seed": DATA_SPLIT_SEED,
        "test_seed": TEST_SEED,
        "beta_L": BETA_L,
        "beta_Q": BETA_Q,
    }

    for key in [
        "f1_global_macro",
        "f1_per_client_macro",
        "mse_macro",
        "ece_macro",
        "mae_macro",
        "ece_prob_macro",
    ]:
        if key in metrics:
            row[key] = metrics[key]

    for risk in RISK_NAMES:
        mse_key = f"mse_risk_{risk}"
        ece_key = f"ece_prob_risk_{risk}"

        if mse_key in metrics:
            row[f"MSE_{risk}"] = metrics[mse_key]

        if ece_key in metrics:
            row[f"ECE_{risk}"] = metrics[ece_key]

    for risk in RISK_NAMES:
        f1_global_key = f"category_global_f1_macro_risk_{risk}"
        f1_pc_key = f"category_per_client_f1_macro_risk_{risk}"

        if f1_global_key in metrics:
            row[f"F1_global_{risk}"] = metrics[f1_global_key]

        if f1_pc_key in metrics:
            row[f"F1_per_client_{risk}"] = metrics[f1_pc_key]

    if "consistency_per_client/fleiss_kappa_macro" in metrics:
        row["fleiss_kappa_per_client_macro"] = metrics[
            "consistency_per_client/fleiss_kappa_macro"
        ]

    if "consistency_global/fleiss_kappa_macro" in metrics:
        row["fleiss_kappa_global_macro"] = metrics[
            "consistency_global/fleiss_kappa_macro"
        ]

    row["partition_final_train_size"] = partition_metadata.get("final_train_size")
    row["partition_test_size"] = partition_metadata.get("test_size")
    row["partition_pool_rows"] = partition_metadata.get("pool_rows")
    row["partition_remaining_rows_after_test_split"] = partition_metadata.get(
        "remaining_rows_after_test_split"
    )

    het = partition_metadata.get("heterogeneity_metrics", {})
    for k, v in het.items():
        row[f"partition_{k}"] = v

    qty = partition_metadata.get("quantity_skew_metrics", {})
    for k, v in qty.items():
        row[f"partition_{k}"] = v

    return row


def run_single_seed(data_seed: int) -> Dict[str, Any]:

    print("\n" + "=" * 80)
    print(f"Running FEDERATED seed sensitivity for data_seed={data_seed}")
    print("=" * 80)

    dataset_path, testset_path, partition_metadata = generate_data_for_seed(data_seed)

    base_cfg = load_base_config()

    cfg = config_for_run(
        base_config=base_cfg,
        data_seed=data_seed,
        dataset_path=dataset_path,
        testset_path=testset_path,
    )

    all_seeds(cfg.model_seed)

    result = run_federated(cfg)

    if isinstance(result, dict) and "summary_metrics" in result:
        metrics = result["summary_metrics"]
    else:
        metrics = result or {}

    return extract_summary_row(
        data_seed=data_seed,
        metrics=metrics,
        partition_metadata=partition_metadata,
    )


def seed_sensitivity() -> None:

    rows: List[Dict[str, Any]] = []

    for data_seed in DATA_SEEDS:
        row = run_single_seed(data_seed)
        rows.append(row)

        df_partial = pd.DataFrame(rows)
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_partial.to_csv(SUMMARY_PATH, index=False)

        if "f1_global_macro" in row:
            print(f"[RESULT] seed={data_seed}, f1_global_macro={row['f1_global_macro']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_PATH, index=False)

    print(f"\nSaved federated seed sensitivity summary to: {SUMMARY_PATH}")

    if "f1_global_macro" in df.columns:
        mean_f1 = df["f1_global_macro"].mean()
        std_f1 = df["f1_global_macro"].std()
        min_f1 = df["f1_global_macro"].min()
        max_f1 = df["f1_global_macro"].max()

        print("\nFederated seed sensitivity summary:")
        print(f"Mean F1-global macro: {mean_f1:.4f}")
        print(f"Std F1-global macro:  {std_f1:.4f}")
        print(f"Min F1-global macro:  {min_f1:.4f}")
        print(f"Max F1-global macro:  {max_f1:.4f}")
        print(f"Range:                {max_f1 - min_f1:.4f}")

        plt.figure(figsize=(8, 5))
        plt.plot(df["data_seed"], df["f1_global_macro"], marker="o", label="F1 global macro")
        plt.axhline(mean_f1, linestyle="--", label=f"Mean = {mean_f1:.4f}")

        plt.xlabel("Data generation seed")
        plt.ylabel("F1 global macro")
        plt.title("Federated seed sensitivity")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_PATH, dpi=300)
        plt.show()

        print(f"Saved plot to: {PLOT_PATH}")


def main() -> None:
    seed_sensitivity()


if __name__ == "__main__":
    main()