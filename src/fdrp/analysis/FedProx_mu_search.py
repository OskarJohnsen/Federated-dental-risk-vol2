from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import copy
import json

import pandas as pd
import numpy as np

from fdrp.ml.config.experiment_config import ExperimentConfig
from fdrp.ml.util.seed import all_seeds
from fdrp.core.paths import root_path, ensure_dir
from fdrp.ml.constants import DATASET, IID_TYPE, RISK_NAMES

from fdrp.data_generation.config.loader import load_all_configs
from fdrp.data_generation.generation.synth import generate_dataset
from fdrp.data_generation.splits import create_global_test_set

from fdrp.ml.centralized.train import main as run_centralized
from fdrp.ml.local.train import main as run_local
from fdrp.ml.federated.train import main as run_federated


# =========================
# USER SETTINGS
# =========================
BETA_L = 0.25
BETA_Q = 1.5
SEEDS = [42]

MU_VALUES = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

RUN_CENTRALIZED = True
RUN_LOCAL = True
RUN_FEDERATED = True

SUMMARY_PATH = Path(r"C:\Users\Oskar\Desktop\fedprox_mu_sweep_summary.csv")


def combo_name(beta_L: float, beta_Q: float, seed: int) -> str:
    return f"betaL_{beta_L}_betaQ_{beta_Q}_seed_{seed}"


# ============================================================
# DATA GENERATION
# ============================================================
def generate_data_for_fixed_combo(
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

    if seed is not None:
        np.random.seed(seed)
        gen_cfg["dataset"]["random_seed"] = seed
    else:
        np.random.seed(gen_cfg["dataset"]["random_seed"])

    part_cfg = gen_cfg.setdefault("partitioning", {})
    part_cfg["beta"] = float(beta_L)

    qty_cfg = part_cfg.setdefault("quantity_skew", {})
    qty_cfg["beta"] = float(beta_Q)

    df, global_thresholds = generate_dataset(configs)
    partition_metadata = global_thresholds.get("_partition_metadata", {})

    cfg_out_dir = gen_cfg["output"]["output_dir"]
    combo = combo_name(beta_L, beta_Q, seed)
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
    df.to_csv(dataset_csv_path, index=False)
    print(f"[DATA] Saved synthetic dataset to: {dataset_csv_path}")

    thresholds_path = (
        proj_root
        / "configs"
        / "global_thresholds"
        / f"{DATASET}"
        / f"global_thresholds_{IID_TYPE}.json"
    )
    ensure_dir(thresholds_path.parent)
    with thresholds_path.open("w") as f:
        json.dump(global_thresholds, f, indent=2)
    print(f"[DATA] Saved global thresholds to: {thresholds_path}")

    test_output_path = proj_root.joinpath(
        "data",
        "processed",
        f"{DATASET}",
        f"global_test_set_{IID_TYPE}_{combo}.csv",
    )
    ensure_dir(test_output_path.parent)

    create_global_test_set(
        dataset_path=dataset_csv_path,
        output_path=test_output_path,
        n_samples=3000,
        seed=999,
        backup_original=True,
    )
    print(f"[DATA] Saved global test set to: {test_output_path}")

    return dataset_csv_path, test_output_path, partition_metadata


# ============================================================
# CONFIG
# ============================================================
def load_base_config(paradigm: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_type=paradigm,
        experiment_id=f"fedprox_mu_sweep_{paradigm}",
    )


def config_for_run(
    base_config: ExperimentConfig,
    paradigm: str,
    beta_L: float,
    beta_Q: float,
    seed: int,
    dataset_path: Path,
    testset_path: Path,
    federated_method: str = "fedavg",
    fedprox_mu: float = 0.0,
) -> ExperimentConfig:
    cfg = copy.deepcopy(base_config)

    cfg.experiment_type = paradigm
    cfg.dataset_path = str(dataset_path)
    cfg.test_set_path = str(testset_path)
    cfg.model_seed = seed

    cfg.category_strategy = "both"
    cfg.threshold_method = "youden"

    cfg.beta_L = float(beta_L)
    cfg.beta_Q = float(beta_Q)

    if paradigm == "federated":
        if getattr(cfg, "federated_rounds", None) is None:
            cfg.federated_rounds = 6
        if getattr(cfg, "local_epochs", None) is None:
            cfg.local_epochs = 5

        cfg.aggregation_method = "fedavg"
        cfg.federated_method = federated_method
        cfg.fedprox_mu = fedprox_mu

    return cfg


# ============================================================
# SUMMARY EXTRACTION
# ============================================================
def extract_summary_row(
    paradigm: str,
    beta_L: float,
    beta_Q: float,
    seed: int,
    metrics: Dict[str, Any],
    federated_method: str | None = None,
    fedprox_mu: float | None = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "paradigm": paradigm,
        "beta_L": beta_L,
        "beta_Q": beta_Q,
        "seed": seed,
        "federated_method": federated_method,
        "fedprox_mu": fedprox_mu,
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
        f1_global_key = f"category_global_f1_macro_risk_{risk}"
        f1_pc_key = f"category_per_client_f1_macro_risk_{risk}"
        mse_key = f"mse_risk_{risk}"
        ece_key = f"ece_prob_risk_{risk}"

        if f1_global_key in metrics:
            row[f"F1_global_{risk}"] = metrics[f1_global_key]
        if f1_pc_key in metrics:
            row[f"F1_per_client_{risk}"] = metrics[f1_pc_key]
        if mse_key in metrics:
            row[f"MSE_{risk}"] = metrics[mse_key]
        if ece_key in metrics:
            row[f"ECE_{risk}"] = metrics[ece_key]

    if "consistency_per_client/patient_disagreement_macro" in metrics:
        row["disagreement_per_client_macro"] = metrics[
            "consistency_per_client/patient_disagreement_macro"
        ]
    if "consistency_global/patient_disagreement_macro" in metrics:
        row["disagreement_global_macro"] = metrics[
            "consistency_global/patient_disagreement_macro"
        ]

    if "consistency_per_client/fleiss_kappa_macro" in metrics:
        row["fleiss_kappa_per_client_macro"] = metrics[
            "consistency_per_client/fleiss_kappa_macro"
        ]
    if "consistency_global/fleiss_kappa_macro" in metrics:
        row["fleiss_kappa_global_macro"] = metrics[
            "consistency_global/fleiss_kappa_macro"
        ]

    return row


def add_partition_metadata(row: Dict[str, Any], partition_metadata: Dict[str, Any]) -> Dict[str, Any]:
    row["partition_beta_L"] = partition_metadata.get("beta")
    row["partition_beta_Q"] = partition_metadata.get("beta_qty")
    row["partition_label"] = partition_metadata.get("label_column")
    row["partition_min_size"] = partition_metadata.get("min_size")

    het = partition_metadata.get("heterogeneity_metrics", {})
    for k, v in het.items():
        row[f"partition_{k}"] = v

    qty = partition_metadata.get("quantity_skew_metrics", {})
    for k, v in qty.items():
        row[f"partition_{k}"] = v

    return row


# ============================================================
# RUNNERS
# ============================================================
def run_single_non_federated(
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
    else:
        raise ValueError(f"Unsupported non-federated paradigm: {paradigm}")

    metrics = result["summary_metrics"] if isinstance(result, dict) and "summary_metrics" in result else (result or {})

    return extract_summary_row(
        paradigm=paradigm,
        beta_L=beta_L,
        beta_Q=beta_Q,
        seed=seed,
        metrics=metrics,
        federated_method=None,
        fedprox_mu=None,
    )


def run_single_federated(
    beta_L: float,
    beta_Q: float,
    seed: int,
    dataset_path: Path,
    testset_path: Path,
    federated_method: str,
    fedprox_mu: float,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(
        f"Running FEDERATED ({federated_method}, mu={fedprox_mu}) "
        f"for beta_L={beta_L}, beta_Q={beta_Q}, seed={seed}"
    )
    print("=" * 80)

    base_cfg = load_base_config("federated")
    cfg = config_for_run(
        base_cfg,
        "federated",
        beta_L,
        beta_Q,
        seed,
        dataset_path,
        testset_path,
        federated_method=federated_method,
        fedprox_mu=fedprox_mu,
    )
    all_seeds(cfg.model_seed)

    result = run_federated(cfg)
    metrics = result["summary_metrics"] if isinstance(result, dict) and "summary_metrics" in result else (result or {})

    return extract_summary_row(
        paradigm="federated",
        beta_L=beta_L,
        beta_Q=beta_Q,
        seed=seed,
        metrics=metrics,
        federated_method=federated_method,
        fedprox_mu=fedprox_mu,
    )


# ============================================================
# MAIN SWEEP
# ============================================================
def fedprox_mu_sweep(
    beta_L: float,
    beta_Q: float,
    seeds: List[int],
    mu_values: List[float],
    summary_path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        dataset_path, testset_path, partition_metadata = generate_data_for_fixed_combo(
            beta_L, beta_Q, seed
        )

        if RUN_CENTRALIZED:
            row = run_single_non_federated(
                "centralized", beta_L, beta_Q, seed, dataset_path, testset_path
            )
            rows.append(add_partition_metadata(row, partition_metadata))

        if RUN_LOCAL:
            row = run_single_non_federated(
                "local", beta_L, beta_Q, seed, dataset_path, testset_path
            )
            rows.append(add_partition_metadata(row, partition_metadata))

        if RUN_FEDERATED:
            for mu in mu_values:
                federated_method = "fedavg" if mu == 0.0 else "fedprox"

                row = run_single_federated(
                    beta_L=beta_L,
                    beta_Q=beta_Q,
                    seed=seed,
                    dataset_path=dataset_path,
                    testset_path=testset_path,
                    federated_method=federated_method,
                    fedprox_mu=mu,
                )
                rows.append(add_partition_metadata(row, partition_metadata))

        df = pd.DataFrame(rows)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_path, index=False)
        print(f"\nIntermediate save to: {summary_path}")

    print(f"\nFinished. Summary saved to: {summary_path}")


def main() -> None:
    fedprox_mu_sweep(
        beta_L=BETA_L,
        beta_Q=BETA_Q,
        seeds=SEEDS,
        mu_values=MU_VALUES,
        summary_path=SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()