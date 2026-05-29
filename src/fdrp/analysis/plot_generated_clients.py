from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdrp.core.paths import root_path, ensure_dir
from fdrp.ml.constants import DATASET, IID_TYPE

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

"""
The code below generates and plots a dataset with required user settings. This code was used to test datapartitioning, which we have changed.
Chat-GPT have assisted during writing of this code.
"""


# ============================================================
# USER SETTINGS
# ============================================================
BETA_L = 1.0
BETA_Q = 1.0
SEED = 1

TEST_SIZE = 3000
MIN_SIZE = 100

PLOT_FULL_DATASET = True   # True = plot train dataset, False = plot global test set
SAVE_PLOT = False


# ============================================================
# DATA GENERATION
# ============================================================
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
        seed=999,
    )

    # 3) partition the remaining dataset directly
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

    # partition metadata
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
    print(f"[DATA] Saved global test set to: {test_output_path}")

    return dataset_csv_path, test_output_path, partition_metadata


# ============================================================
# PLOTTING
# ============================================================
def plot_patients_per_client_with_marked_complications(
    df: pd.DataFrame,
    client_col: str = "Client",
    title: str = "Patients per Client with Marked Complications",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot per client:
    - number with at least one complication
    - number with no complications
    """
    df = df.copy()

    if client_col not in df.columns:
        raise ValueError(f"Column '{client_col}' not found in dataframe.")

    complication_cols = [
        "Risk_AlveolarOsteitis",
        "Risk_SecondaryInfection",
        "Risk_NerveDysesthesia",
        "Risk_Bleeding",
    ]

    missing = [c for c in complication_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing complication columns: {missing}")

    df["AtLeastOneComplication"] = (df[complication_cols].max(axis=1) > 0).astype(int)

    summary = (
        df.groupby(client_col)["AtLeastOneComplication"]
        .agg(with_complication="sum", total="count")
        .reset_index()
    )

    summary["without_complication"] = summary["total"] - summary["with_complication"]
    summary["pct_with_complication"] = 100 * summary["with_complication"] / summary["total"]

    try:
        summary = summary.sort_values(client_col, key=lambda s: pd.to_numeric(s))
    except Exception:
        summary = summary.sort_values(client_col)

    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x,
        summary["with_complication"],
        label="At least one complication",
    )

    ax.bar(
        x,
        summary["without_complication"],
        bottom=summary["with_complication"],
        label="No complication",
    )

    y_offset = summary["total"].max() * 0.01

    for i, row in enumerate(summary.itertuples(index=False)):
        ax.text(
            i,
            row.total + y_offset,
            f"{int(row.with_complication)} ({row.pct_with_complication:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(summary[client_col])
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Patients")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved figure to: {save_path}")

    plt.show()


def plot_four_complications_per_client(
    df: pd.DataFrame,
    client_col: str = "Client",
    save_path: str | Path | None = None,
    overall_title: str = "Complications per Client",
) -> None:
    """
    Create one figure with 4 small plots, one for each complication.
    Each subplot shows:
    - number with complication
    - number without complication
    - annotation with count and percentage with complication
    """
    df = df.copy()

    if client_col not in df.columns:
        raise ValueError(f"Column '{client_col}' not found in dataframe.")

    complication_info = [
        ("Risk_AlveolarOsteitis", "Alveolar Osteitis"),
        ("Risk_SecondaryInfection", "Secondary Infection"),
        ("Risk_NerveDysesthesia", "Nerve Dysesthesia"),
        ("Risk_Bleeding", "Bleeding"),
    ]

    missing = [col for col, _ in complication_info if col not in df.columns]
    if missing:
        raise ValueError(f"Missing complication columns: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for ax, (col, pretty_name) in zip(axes, complication_info):
        summary = (
            df.groupby(client_col)[col]
            .agg(with_complication="sum", total="count")
            .reset_index()
        )

        summary["without_complication"] = summary["total"] - summary["with_complication"]
        summary["pct_with_complication"] = 100 * summary["with_complication"] / summary["total"]

        try:
            summary = summary.sort_values(client_col, key=lambda s: pd.to_numeric(s))
        except Exception:
            summary = summary.sort_values(client_col)

        x = np.arange(len(summary))

        ax.bar(
            x,
            summary["with_complication"],
            label="Complication",
        )
        ax.bar(
            x,
            summary["without_complication"],
            bottom=summary["with_complication"],
            label="No complication",
        )

        y_offset = summary["total"].max() * 0.01

        for i, row in enumerate(summary.itertuples(index=False)):
            ax.text(
                i,
                row.total + y_offset,
                f"{int(row.with_complication)} ({row.pct_with_complication:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(summary[client_col])
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Number of Patients")
        ax.set_title(pretty_name)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(overall_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved figure to: {save_path}")

    plt.show()

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    dataset_path, testset_path, partition_metadata = generate_data_for_beta_combo(
        beta_L=BETA_L,
        beta_Q=BETA_Q,
        seed=SEED,
    )

    chosen_path = dataset_path if PLOT_FULL_DATASET else testset_path
    df = pd.read_csv(chosen_path)

    print(f"[INFO] Plotting data from: {chosen_path}")
    print(f"[INFO] Partition metadata: {partition_metadata}")

    atleast_one_save_path = None
    four_complications_save_path = None

    if SAVE_PLOT:
        proj_root = root_path()
        plot_dir = proj_root / "src" / "fdrp" / "analysis" / "plots"
        ensure_dir(plot_dir)

        data_tag = "train_dataset" if PLOT_FULL_DATASET else "global_test_set"

        atleast_one_save_path = plot_dir / (
            f"patients_per_client_atleast_one_{data_tag}_betaL_{BETA_L}_betaQ_{BETA_Q}_seed_{SEED}.png"
        )

        four_complications_save_path = plot_dir / (
            f"patients_per_client_four_complications_{data_tag}_betaL_{BETA_L}_betaQ_{BETA_Q}_seed_{SEED}.png"
        )

    data_name = "Train dataset" if PLOT_FULL_DATASET else "Global test set"

    title_atleast_one = (
        f"Patients per Client with Marked Complications\n"
        f"({data_name}, beta_L={BETA_L}, beta_Q={BETA_Q}, seed={SEED})"
    )

    title_four = (
        f"Complications per Client\n"
        f"({data_name}, beta_L={BETA_L}, beta_Q={BETA_Q}, seed={SEED})"
    )

    plot_patients_per_client_with_marked_complications(
        df=df,
        client_col="Client",
        title=title_atleast_one,
        save_path=atleast_one_save_path,
    )

    plot_four_complications_per_client(
        df=df,
        client_col="Client",
        save_path=four_complications_save_path,
        overall_title=title_four,
    )


if __name__ == "__main__":
    main()