from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, List
import itertools
import copy
import json

import pandas as pd
import numpy as np

from fdrp.ml.config.experiment_config import ExperimentConfig
from fdrp.ml.util.seed import all_seeds
from fdrp.core.paths import root_path, ensure_dir
from fdrp.ml.constants import DATASET, IID_TYPE, RISK_NAMES

# === Datagenerering ==========================================================
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

# === ML-scripts ==============================================================
from fdrp.ml.centralized.train import main as run_centralized
from fdrp.ml.local.train import main as run_local
from fdrp.ml.federated.train import main as run_federated


# Grid
BETA_L_VALUES = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,50.0]
BETA_Q_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 50.0]
PARADIGMS = ["centralized", "local", "federated"]

SUMMARY_PATH = Path(r"C:\Users\Oskar\Desktop\sweep_beta_summary.csv")

TEST_SIZE = 3000
MIN_SIZE = 100
TEST_SEED = 999


def beta_combo_name(beta_L: float, beta_Q: float) -> str:
    return f"betaL_{beta_L}_betaQ_{beta_Q}"


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_data_for_beta_combo(
    beta_L: float,
    beta_Q: float,
    seed: int,
) -> tuple[Path, Path, Dict[str, Any]]:

    print(f"\n[DATA] Generating dataset for beta_L={beta_L}, beta_Q={beta_Q}, seed={seed}")

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
    pool_cfg["beta_L"] = float(beta_L)
    pool_cfg["beta_Q"] = float(beta_Q)
    pool_cfg["min_size"] = int(MIN_SIZE)

    # 1) generate full dataset
    df_pool, global_thresholds = generate_dataset(configs)

    # 2) split off global test set
    df_remaining_pool, df_test = split_global_test_from_pool(
        df=df_pool,
        n_test_samples=TEST_SIZE,
        seed=TEST_SEED,
    )

    # 3) partition remaining data directly
    n_clients = gen_cfg["dataset"]["n_clients"]

    df_train = partition_dataset_constrained_dirichlet(
        df=df_remaining_pool,
        n_clients=n_clients,
        beta_L=beta_L,
        beta_Q=beta_Q,
        label_column="Risk_Category_Composite",
        client_column="Client",
        min_size=MIN_SIZE,
        seed=seed,
    )

    # metrics / metadata
    print_partition_statistics(df_train, "Risk_Category_Composite", "Client")

    heterogeneity_metrics = compute_partition_heterogeneity_metrics(
        df_train, "Risk_Category_Composite", "Client"
    )

    print("\nPartition Heterogeneity Metrics")
    for metric_name, value in heterogeneity_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print_quantity_skew_statistics(df_train, "Client")

    quantity_metrics = compute_quantity_skew_metrics(df_train, "Client")

    print("\nQuantity Skew Metrics")
    for metric_name, value in quantity_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    partition_metadata = {
        "beta_L": beta_L,
        "beta_Q": beta_Q,
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

    # save outputs
    cfg_out_dir = gen_cfg["output"]["output_dir"]
    combo = f"betaL_{beta_L}_betaQ_{beta_Q}_seed_{seed}"
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

    try:
        print(f"[DATA] Saved train dataset to: {dataset_csv_path.relative_to(proj_root)}")
    except Exception:
        print(f"[DATA] Saved train dataset to: {dataset_csv_path}")

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

    try:
        print(f"[DATA] Saved thresholds to: {thresholds_path.relative_to(proj_root)}")
    except Exception:
        print(f"[DATA] Saved thresholds to: {thresholds_path}")

    test_output_path = (
        proj_root
        / "data"
        / "processed"
        / f"{DATASET}"
        / f"global_test_set_{IID_TYPE}_{combo}.csv"
    )
    ensure_dir(test_output_path.parent)

    df_test.to_csv(test_output_path, index=False)

    try:
        print(f"[DATA] Saved global test set to: {test_output_path.relative_to(proj_root)}")
    except Exception:
        print(f"[DATA] Saved global test set to: {test_output_path}")

    return dataset_csv_path, test_output_path, partition_metadata


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_base_config(paradigm: str) -> ExperimentConfig:

    return ExperimentConfig(
        experiment_type=paradigm,
        experiment_id=f"beta_sweep_{paradigm}",
    )


def config_for_run(
    base_config: ExperimentConfig,
    paradigm: str,
    beta_L: float,
    beta_Q: float,
    seed: int,
    dataset_path: Path,
    testset_path: Path,
) -> ExperimentConfig:

    cfg = copy.deepcopy(base_config)

    cfg.experiment_type = paradigm
    cfg.dataset_path = str(dataset_path)
    cfg.test_set_path = str(testset_path)
    cfg.model_seed = seed
    cfg.data_split_seed = seed

    cfg.category_strategy = "both"
    cfg.threshold_method = "percentile"
    cfg.use_wandb = False

    cfg.beta_L = float(beta_L)
    cfg.beta_Q = float(beta_Q)
    cfg.run_suffix = beta_combo_name(beta_L, beta_Q)

    if paradigm == "federated":
        if getattr(cfg, "federated_rounds", None) is None:
            cfg.federated_rounds = 6

        if getattr(cfg, "local_epochs", None) is None:
            cfg.local_epochs = 5

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# METRICS EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_summary_row(
    paradigm: str,
    beta_L: float,
    beta_Q: float,
    seed: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:

    row: Dict[str, Any] = {
        "paradigm": paradigm,
        "beta_L": beta_L,
        "beta_Q": beta_Q,
        "seed": seed,
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
        row["fleiss_kappa_per_client_macro"] = metrics["consistency_per_client/fleiss_kappa_macro"]

    if "consistency_global/fleiss_kappa_macro" in metrics:
        row["fleiss_kappa_global_macro"] = metrics["consistency_global/fleiss_kappa_macro"]

    return row


# ─────────────────────────────────────────────────────────────────────────────
# RUN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(
    paradigm: str,
    beta_L: float,
    beta_Q: float,
    seed: int,
    dataset_path: Path,
    testset_path: Path,
) -> Dict[str, Any]:

    print("\n" + "=" * 80)
    print(f"Running {paradigm.upper()} for beta_L={beta_L}, beta_Q={beta_Q}, seed={seed}")
    print("=" * 80)

    base_cfg = load_base_config(paradigm)

    cfg = config_for_run(
        base_cfg,
        paradigm,
        beta_L,
        beta_Q,
        seed,
        dataset_path,
        testset_path,
    )

    all_seeds(cfg.model_seed)

    if paradigm == "centralized":
        result = run_centralized(cfg)
    elif paradigm == "local":
        result = run_local(cfg)
    elif paradigm == "federated":
        result = run_federated(cfg)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")

    if isinstance(result, dict) and "summary_metrics" in result:
        metrics = result["summary_metrics"]
    else:
        metrics = result or {}

    return extract_summary_row(paradigm, beta_L, beta_Q, seed, metrics)


# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def beta_sweep(
    beta_L_values: Iterable[float],
    beta_Q_values: Iterable[float],
    paradigms: Iterable[str],
    seeds: Iterable[int],
    summary_path: Path,
) -> None:

    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        for beta_L, beta_Q in itertools.product(beta_L_values, beta_Q_values):

            dataset_path, testset_path, partition_metadata = generate_data_for_beta_combo(
                beta_L,
                beta_Q,
                seed,
            )

            for paradigm in paradigms:

                row = run_single_experiment(
                    paradigm,
                    beta_L,
                    beta_Q,
                    seed,
                    dataset_path,
                    testset_path,
                )

                # Save partition metadata
                row["partition_beta_L"] = partition_metadata.get("beta_L")
                row["partition_beta_Q"] = partition_metadata.get("beta_Q")
                row["partition_label"] = partition_metadata.get("label_column")
                row["partition_min_size"] = partition_metadata.get("min_size")
                row["partition_method"] = partition_metadata.get("partition_method")
                row["partition_final_train_size"] = partition_metadata.get("final_train_size")
                row["partition_test_size"] = partition_metadata.get("test_size")
                row["partition_pool_rows"] = partition_metadata.get("pool_rows")
                row["partition_remaining_rows_after_test_split"] = partition_metadata.get(
                    "remaining_rows_after_test_split"
                )

                # heterogeneity metrics
                het = partition_metadata.get("heterogeneity_metrics", {})
                for k, v in het.items():
                    row[f"partition_{k}"] = v

                # quantity skew metrics
                qty = partition_metadata.get("quantity_skew_metrics", {})
                for k, v in qty.items():
                    row[f"partition_{k}"] = v

                rows.append(row)

                df_partial = pd.DataFrame(rows)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                df_partial.to_csv(summary_path, index=False)

    df = pd.DataFrame(rows)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)

    print(f"\nSaved sweep summary to: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    seeds = [42]

    beta_sweep(
        beta_L_values=BETA_L_VALUES,
        beta_Q_values=BETA_Q_VALUES,
        paradigms=PARADIGMS,
        seeds=seeds,
        summary_path=SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()