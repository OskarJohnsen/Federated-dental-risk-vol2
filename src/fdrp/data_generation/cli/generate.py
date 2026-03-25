from __future__ import annotations
from pathlib import Path
import json
import typer
import numpy as np

from ..config.loader import load_all_configs
from ..generation.synth import generate_dataset
from ..splits import split_global_test_from_pool
from ..partitioning.pool_partitioning import (
    partition_dataset_constrained_dirichlet,
    print_partition_statistics,
    compute_partition_heterogeneity_metrics,
    print_quantity_skew_statistics,
    compute_quantity_skew_metrics,
)
from ...core.paths import ensure_dir, root_path
from fdrp.ml.constants import DATASET, IID_TYPE

app = typer.Typer(add_completion=False)


@app.command()
def main(
    seed: int | None = typer.Option(None, help="Random seed override"),
    output_dir: Path | None = typer.Option(None, help="Output directory"),
    formats: str = typer.Option("csv", help="Formats: csv"),
    create_test_set: bool = typer.Option(True, help="Create global test set"),
    test_samples: int = typer.Option(3000, help="Number of test samples"),
    test_seed: int = typer.Option(999, help="Seed for test split"),
    beta: float = typer.Option(0.5, help="Label skew (fallback)"),
    beta_qty: float = typer.Option(0.7, help="Quantity skew (fallback)"),
):
  

    # ------------------------------------------------------------
    # CONFIG + SEED
    # ------------------------------------------------------------
    configs = load_all_configs()
    configs["iid_type"] = IID_TYPE

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(configs["generation"]["dataset"]["random_seed"])

    # ------------------------------------------------------------
    # STEP 1: GENERATE LARGE POOL
    # ------------------------------------------------------------
    df_pool, global_thresholds = generate_dataset(configs)

    # ------------------------------------------------------------
    # READ CONFIG
    # ------------------------------------------------------------
    gen = configs["generation"]
    pool_cfg = gen.get("pool_partitioning", {})

    n_clients = gen["dataset"]["n_clients"]
    random_seed = gen["dataset"].get("random_seed", 42)

    test_size = pool_cfg.get("test_size", test_samples)
    label_column = pool_cfg.get("label_column", "Risk_Category_Composite")
    beta_L = pool_cfg.get("beta_L", beta)
    beta_Q = pool_cfg.get("beta_Q", beta_qty)
    min_size = pool_cfg.get("min_size", 100)

    # ------------------------------------------------------------
    # OUTPUT PATHS
    # ------------------------------------------------------------
    cfg_out_dir = configs["generation"]["output"]["output_dir"]
    base = f"synthetic_dataset_{DATASET}_{IID_TYPE}"
    proj_root = root_path()

    if output_dir is not None:
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        p = Path(cfg_out_dir)
        out_dir = (proj_root.joinpath(p) if not p.is_absolute() else p).resolve()

        try:
            inside_repo = proj_root == out_dir or proj_root in out_dir.parents
        except Exception:
            inside_repo = False

        if not inside_repo:
            out_dir = proj_root.joinpath("data", "raw").resolve()

    ensure_dir(out_dir)

    fmts = [f.strip() for f in formats.split(",") if f.strip()]

    # ------------------------------------------------------------
    # STEP 2: SPLIT GLOBAL TEST SET
    # ------------------------------------------------------------
    df_remaining_pool, df_test = split_global_test_from_pool(
        df=df_pool,
        n_test_samples=test_size,
        seed=test_seed,
    )

    # ------------------------------------------------------------
    # STEP 3: PARTITION REMAINING DATASET DIRECTLY
    # ------------------------------------------------------------
    df_train = partition_dataset_constrained_dirichlet(
        df=df_remaining_pool,
        n_clients=n_clients,
        beta_L=beta_L,
        beta_Q=beta_Q,
        label_column=label_column,
        client_column="Client",
        min_size=min_size,
        seed=random_seed,
    )
    

    # ------------------------------------------------------------
    # DEBUG / LOGGING
    # ------------------------------------------------------------
    print_partition_statistics(df_train, label_column, "Client")

    heterogeneity_metrics = compute_partition_heterogeneity_metrics(
        df_train, label_column, "Client"
    )

    print("\nPartition Heterogeneity Metrics")
    for k, v in heterogeneity_metrics.items():
        print(f"{k}: {v:.4f}")

    print_quantity_skew_statistics(df_train, "Client")

    quantity_metrics = compute_quantity_skew_metrics(df_train, "Client")

    print("\nQuantity Skew Metrics")
    for k, v in quantity_metrics.items():
        print(f"{k}: {v:.4f}")

    # ------------------------------------------------------------
    # SAVE PARTITION METADATA
    # ------------------------------------------------------------
    global_thresholds["_partition_metadata"] = {
        "beta_L": beta_L,
        "beta_Q": beta_Q,
        "label_column": label_column,
        "iid_type": IID_TYPE,
        "heterogeneity_metrics": heterogeneity_metrics,
        "quantity_skew_metrics": quantity_metrics,
        "final_train_size": int(len(df_train)),
        "test_size": test_size,
    }

    # ------------------------------------------------------------
    # SAVE TRAIN DATASET
    # ------------------------------------------------------------
    train_path = out_dir.joinpath(f"{base}.csv")

    if "csv" in fmts:
        df_train.to_csv(train_path, index=False)

    print(f"\nSaved train dataset: {train_path}")
    print(f"Train samples: {len(df_train):,}")

    # ------------------------------------------------------------
    # SAVE GLOBAL TEST SET
    # ------------------------------------------------------------
    test_output_path = proj_root.joinpath(
        "data", "processed", f"{DATASET}", f"global_test_set_{IID_TYPE}.csv"
    )

    ensure_dir(test_output_path.parent)

    if create_test_set and "csv" in fmts:
        df_test.to_csv(test_output_path, index=False)

    print(f"Saved global test set: {test_output_path}")
    print(f"Test samples: {len(df_test):,}")

    # ------------------------------------------------------------
    # SAVE THRESHOLDS
    # ------------------------------------------------------------
    thresholds_path = proj_root.joinpath(
        "configs", "global_thresholds", f"{DATASET}", f"global_thresholds_{IID_TYPE}.json"
    )

    ensure_dir(thresholds_path.parent)

    with thresholds_path.open("w") as f:
        json.dump(global_thresholds, f, indent=2)

    print(f"Saved thresholds: {thresholds_path}")

    # ------------------------------------------------------------
    # META OUTPUT
    # ------------------------------------------------------------
    meta = {
        "train_rows": int(df_train.shape[0]),
        "test_rows": int(df_test.shape[0]),
        "pool_rows": int(df_pool.shape[0]),
        "n_clients": n_clients,
    }

    typer.echo(json.dumps({"status": "ok", "meta": meta}, indent=2))


def run():
    app()


if __name__ == "__main__":
    run()