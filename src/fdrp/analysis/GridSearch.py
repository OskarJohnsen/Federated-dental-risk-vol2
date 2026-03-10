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

# === Datagenerering: brug samme komponenter som dit generate-script ==========

# Tilpas evt. sti, men dette matcher din generate.py
from fdrp.data_generation.config.loader import load_all_configs
from fdrp.data_generation.generation.synth import generate_dataset
from fdrp.data_generation.splits import create_global_test_set

# === ML-scripts ==============================================================
from fdrp.ml.centralized.train import main as run_centralized
from fdrp.ml.local.train import main as run_local
from fdrp.ml.federated.train import main as run_federated

# Grid:
BETA_L_VALUES = [0.1,0.5,1.0,1.5,2.0, 5.0, 10.0]
BETA_Q_VALUES = [0.1,0.5,1.0,1.5,2.0, 5.0, 10.0]

PARADIGMS = ["centralized", "local", "federated"]

SUMMARY_PATH = Path(r"C:\Users\Oskar\Desktop\sweep_beta_summary.csv")



def beta_combo_name(beta_L: float, beta_Q: float) -> str:
    return f"betaL_{beta_L}_betaQ_{beta_Q}"


# ──────────────────────────────────────────────────────────────────────────────
#  DATA-GENERERING (samme logik som generate.main, men med beta_L / beta_Q)
# ──────────────────────────────────────────────────────────────────────────────

def generate_data_for_beta_combo(
    beta_L: float,
    beta_Q: float,
    seed: int,
) -> tuple[Path, Path]:
    """
    Generér syntetisk dataset + global testset for en given (beta_L, beta_Q, seed).

    - Loader configs via load_all_configs()
    - Sætter iid_type
    - Overskriver:
        - generation.dataset.random_seed = seed
        - generation.partitioning.beta = beta_L
        - generation.partitioning.quantity_skew.beta = beta_Q
    - Kører generate_dataset(configs)
    - Gemmer:
        - synthetic_dataset_{DATASET}_{IID_TYPE}.csv
        - configs/global_thresholds/{DATASET}/global_thresholds_{IID_TYPE}.json
        - data/processed/{DATASET}/global_test_set_{IID_TYPE}.csv

    Returnerer:
        (dataset_csv_path, test_output_path)
    """
    print(f"\n[DATA] Generating dataset for beta_L={beta_L}, beta_Q={beta_Q}, seed={seed}")

    # Load og nulstil cache for hvert sweep-step
    configs = load_all_configs(force_reload=True)

    # Match dit generate-script:
    configs["iid_type"] = IID_TYPE

    gen_cfg = configs["generation"]

    # Seed: bruges både til numpy i generate.main og til random_seed i config
    if seed is not None:
        np.random.seed(seed)
        gen_cfg["dataset"]["random_seed"] = seed
    else:
        np.random.seed(gen_cfg["dataset"]["random_seed"])

    # Sæt label-skew beta (Dirichlet)
    part_cfg = gen_cfg.setdefault("partitioning", {})
    part_cfg["beta"] = float(beta_L)

    # Sæt quantity-skew beta (hvis quantity_skew findes / oprettes)
    qty_cfg = part_cfg.setdefault("quantity_skew", {})
    qty_cfg["beta"] = float(beta_Q)

    # Kald selve generatoren
    df, global_thresholds = generate_dataset(configs)

    # --- Resten følger dit generate.main ---
    cfg_out_dir = gen_cfg["output"]["output_dir"]
    base = f"synthetic_dataset_{DATASET}_{IID_TYPE}"
    proj_root = root_path()

    # Resolve output-dir relativt til projekt-root, ligesom i generate.main
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

    # Gem CSV (vi behøver ikke xlsx til sweep)
    dataset_csv_path = out_dir.joinpath(f"{base}.csv")
    df.to_csv(dataset_csv_path, index=False)
    print(f"[DATA] Saved synthetic dataset to: {dataset_csv_path.relative_to(proj_root)}")

    # Gem global thresholds
    configs_dir = proj_root.joinpath("configs")
    ensure_dir(configs_dir)
    thresholds_path = configs_dir / "global_thresholds" / f"{DATASET}" / f"global_thresholds_{IID_TYPE}.json"
    ensure_dir(thresholds_path.parent)
    with thresholds_path.open("w") as f:
        json.dump(global_thresholds, f, indent=2)
    print(f"[DATA] Saved global thresholds to: {thresholds_path.relative_to(proj_root)}")

    # Global test set
    test_output_path = proj_root.joinpath("data", "processed", f"{DATASET}", f"global_test_set_{IID_TYPE}.csv")
    ensure_dir(test_output_path.parent)

    try:
        from_samples = df.shape[0]
        print(f"[DATA] Creating global test set ({from_samples} rows in full dataset)...")
        create_global_test_set(
            dataset_path=dataset_csv_path,
            output_path=test_output_path,
            n_samples=3000,       # du kan evt. gøre dette til en konstant / parameter
            seed=999,             # eller knytte til 'seed' hvis du vil
            backup_original=True,
        )
        print(f"[DATA] Saved global test set to: {test_output_path.relative_to(proj_root)}")
    except Exception as e:
        print(f"[DATA] Warning: Failed to create test set: {e}")
        print("[DATA] Dataset generation completed, but test set creation failed.")

    return dataset_csv_path, test_output_path


# ──────────────────────────────────────────────────────────────────────────────
#  ML-KONFIG OG SWEEP
# ──────────────────────────────────────────────────────────────────────────────

def load_base_config(paradigm: str) -> ExperimentConfig:
    """
    Lav en simpel 'base' ExperimentConfig.
    Vi overskriver alligevel de fleste felter i config_for_run().
    """
    cfg = ExperimentConfig(
        experiment_type=paradigm,
        experiment_id=f"beta_sweep_{paradigm}",
    )
    return cfg


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

    cfg.category_strategy = "both"
    cfg.threshold_method = "percentile"
    cfg.use_wandb = False

    # Annoteringer til senere analyse
    cfg.beta_L = float(beta_L)
    cfg.beta_Q = float(beta_Q)
    cfg.run_suffix = beta_combo_name(beta_L, beta_Q)

    # 🔹 Federated-specifikke felter
    if paradigm == "federated":
        # vælg nogle fornuftige defaults – dem kan du altid tweake
        if getattr(cfg, "federated_rounds", None) is None:
            cfg.federated_rounds = 6   
        if getattr(cfg, "local_epochs", None) is None:
            cfg.local_epochs = 5       

    return cfg


def extract_summary_row(
    paradigm: str,
    beta_L: float,
    beta_Q: float,
    seed: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Bygger én række til summary.csv ud fra metrics-dict'et.
    Her vælger vi F1, MSE, ECE og Fleiss κ (macro).
    """
    row: Dict[str, Any] = {
        "paradigm": paradigm,
        "beta_L": beta_L,
        "beta_Q": beta_Q,
        "seed": seed,
    }

    # --- 1) Macro-metrics ---------------------------------------------------
    for key in ["f1_global_macro", "f1_per_client_macro", "mse_macro", "ece_macro", "mae_macro", "ece_prob_macro"]:
        if key in metrics:
            row[key] = metrics[key]

    # Hvis du hellere vil have pr. risiko:
    for risk in RISK_NAMES:
        mse_key = f"mse_risk_{risk}"
        ece_key = f"ece_prob_risk_{risk}"
        if mse_key in metrics:
            row[f"MSE_{risk}"] = metrics[mse_key]
        if ece_key in metrics:
            row[f"ECE_{risk}"] = metrics[ece_key]

    # --- 2) F1 global og per klient -----------------------------------------
    for risk in RISK_NAMES:
        f1_global_key = f"category_global_f1_macro_risk_{risk}"
        if f1_global_key in metrics:
            row[f"F1_global_{risk}"] = metrics[f1_global_key]

        f1_pc_key = f"category_per_client_f1_macro_risk_{risk}"
        if f1_pc_key in metrics:
            row[f"F1_per_client_{risk}"] = metrics[f1_pc_key]

    # --- 3) Konsistens-metrics: Fleiss κ + disagreement ---------------------

    # patient disagreement
    if "consistency_per_client/patient_disagreement_macro" in metrics:
        row["disagreement_per_client_macro"] = metrics[
            "consistency_per_client/patient_disagreement_macro"
        ]
    if "consistency_global/patient_disagreement_macro" in metrics:
        row["disagreement_global_macro"] = metrics[
            "consistency_global/patient_disagreement_macro"
        ]

    # Fleiss kappa (macro)
    if "consistency_per_client/fleiss_kappa_macro" in metrics:
        row["fleiss_kappa_per_client_macro"] = metrics[
            "consistency_per_client/fleiss_kappa_macro"
        ]
    if "consistency_global/fleiss_kappa_macro" in metrics:
        row["fleiss_kappa_global_macro"] = metrics[
            "consistency_global/fleiss_kappa_macro"
        ]

    return row


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
    cfg = config_for_run(base_cfg, paradigm, beta_L, beta_Q, seed, dataset_path, testset_path)
    all_seeds(cfg.model_seed)

    if paradigm == "centralized":
        result = run_centralized(cfg)
    elif paradigm == "local":
        result = run_local(cfg)
    elif paradigm == "federated":
        result = run_federated(cfg)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")

    # Hvis result er nested (fx local), så vælg summary-dict
    if isinstance(result, dict) and "summary_metrics" in result:
        metrics = result["summary_metrics"]
    else:
        metrics = result or {}

    row = extract_summary_row(paradigm, beta_L, beta_Q, seed, metrics)
    return row


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
            # 1) Generér data én gang per (beta_L, beta_Q, seed)
            dataset_path, testset_path = generate_data_for_beta_combo(beta_L, beta_Q, seed)

            # 2) Kør alle paradigmer på det dataset
            for paradigm in paradigms:
                row = run_single_experiment(paradigm, beta_L, beta_Q, seed, dataset_path, testset_path)
                rows.append(row)

    df = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"\nSaved sweep summary to: {summary_path}")


def main() -> None:
    seeds = [42]  # du kan udvide senere
    beta_sweep(
        beta_L_values=BETA_L_VALUES,
        beta_Q_values=BETA_Q_VALUES,
        paradigms=PARADIGMS,
        seeds=seeds,
        summary_path=SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()