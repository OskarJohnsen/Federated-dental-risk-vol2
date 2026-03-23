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
from fdrp.data_generation.splits import create_global_test_set


# ============================================================
# USER SETTINGS
# ============================================================
BETA_L = 100.0
BETA_Q = 5.0
SEED = 50

PLOT_FULL_DATASET = True
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

    part_cfg = gen_cfg.setdefault("partitioning", {})
    part_cfg["beta"] = float(beta_L)

    qty_cfg = part_cfg.setdefault("quantity_skew", {})
    qty_cfg["beta"] = float(beta_Q)

    df, global_thresholds = generate_dataset(configs)
    partition_metadata = global_thresholds.get("_partition_metadata", {})

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

    test_output_path = (
        proj_root
        / "data"
        / "processed"
        / f"{DATASET}"
        / f"global_test_set_{IID_TYPE}_{combo}.csv"
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
# PLOTTING
# ============================================================
def plot_patients_per_client_with_marked_complications(
    df: pd.DataFrame,
    client_col: str = "Client",
    title: str = "Patients per Client with Marked Complications",
    save_path: str | Path | None = None,
) -> None:
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

    df["AtLeastOneComplication"] = (
        df[complication_cols].max(axis=1) > 0
    ).astype(int)

    summary = (
        df.groupby(client_col)["AtLeastOneComplication"]
        .agg(with_complication="sum", total="count")
        .reset_index()
    )

    summary["without_complication"] = summary["total"] - summary["with_complication"]
    summary["pct_with_complication"] = (
        100 * summary["with_complication"] / summary["total"]
    )

    try:
        summary = summary.sort_values(client_col, key=lambda s: pd.to_numeric(s))
    except Exception:
        summary = summary.sort_values(client_col)

    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x,
        summary["with_complication"],
        color="orange",
        label="At least one complication",
    )

    ax.bar(
        x,
        summary["without_complication"],
        bottom=summary["with_complication"],
        color="blue",
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

    save_path = None
    if SAVE_PLOT:
        proj_root = root_path()
        plot_dir = proj_root / "src" / "fdrp" / "analysis" / "plots"
        ensure_dir(plot_dir)

        data_tag = "full_dataset" if PLOT_FULL_DATASET else "global_test_set"
        save_path = plot_dir / (
            f"patients_per_client_{data_tag}_betaL_{BETA_L}_betaQ_{BETA_Q}_seed_{SEED}.png"
        )

    data_name = "Full generated dataset" if PLOT_FULL_DATASET else "Global test set"
    title = (
        f"Patients per Client with Marked Complications\n"
        f"({data_name}, beta_L={BETA_L}, beta_Q={BETA_Q}, seed={SEED})"
    )

    plot_patients_per_client_with_marked_complications(
        df=df,
        client_col="Client",
        title=title,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()