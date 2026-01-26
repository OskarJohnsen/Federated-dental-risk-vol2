from __future__ import annotations
from pathlib import Path
import json
import typer
import numpy as np
from ..config.loader import load_all_configs
from ..generation.synth import generate_dataset
from ..splits import create_global_test_set
from ...core.paths import ensure_dir, root_path
from digit_fr.ml.constants import DATASET, IID_TYPE

app = typer.Typer(add_completion=False)

@app.command()
def main(
    seed: int | None = typer.Option(None, help="Random seed override"),
    output_dir: Path | None = typer.Option(None, help="Output directory; defaults to config output.output_dir from project root"),
    formats: str = typer.Option("csv,xlsx", help="Comma-separated formats: csv,xlsx"),
    create_test_set: bool = typer.Option(True, help="Create global test set automatically"),
    test_samples: int = typer.Option(3000, help="Number of samples in test set"),
    test_seed: int = typer.Option(999, help="Random seed for test set splitting"),
    backup: bool = typer.Option(True, help="Create backup of original dataset"),
):
    """Generate the synthetic dataset and optionally create a global test set"""
    configs = load_all_configs()
    configs["iid_type"] = IID_TYPE
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(configs["generation"]["dataset"]["random_seed"])

    df, global_thresholds = generate_dataset(configs)

    cfg_out_dir = configs["generation"]["output"]["output_dir"]
    base = f"fed_recommenders_synthetic_dataset_{DATASET}_{IID_TYPE}"
    proj_root = root_path()
    # Resolve output directory
    if output_dir is not None:
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        p = Path(cfg_out_dir)
        # Resolve relative to project root
        out_dir = (proj_root.joinpath(p) if not p.is_absolute() else p).resolve()
        try:
            inside_repo = proj_root == out_dir or proj_root in out_dir.parents
        except Exception:
            inside_repo = False
        if not inside_repo:
            out_dir = proj_root.joinpath("data", "raw").resolve()
    ensure_dir(out_dir)

    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    if "csv" in fmts:
        df.to_csv(out_dir.joinpath(f"{base}.csv"), index=False)

    configs_dir = proj_root.joinpath("configs")
    ensure_dir(configs_dir)
    thresholds_path = configs_dir / "global_thresholds" / f"{DATASET}" / f"global_thresholds_{IID_TYPE}.json"
    print(f"\nSaving global thresholds: {thresholds_path}")
    ensure_dir(thresholds_path.parent)
    with thresholds_path.open("w") as f:
        json.dump(global_thresholds, f, indent=2)

    # Create global test set if requested
    dataset_csv_path = out_dir.joinpath(f"{base}.csv")
    if create_test_set and "csv" in fmts:
        test_output_path = proj_root.joinpath("data", "processed", f"{DATASET}", f"global_test_set_{IID_TYPE}.csv")
        try:
            create_global_test_set(
                dataset_path=dataset_csv_path,
                output_path=test_output_path,
                n_samples=test_samples,
                seed=test_seed,
                backup_original=backup
            )
        except Exception as e:
            typer.echo(f"Warning: Failed to create test set: {e}", err=True)
            typer.echo("Dataset generation completed, but test set creation failed.", err=True)

    meta = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "formats": fmts,
        "output_dir": str(out_dir.resolve()),
        "test_set_created": create_test_set and "csv" in fmts,
    }
    
    if "_partition_metadata" in global_thresholds:
        meta["partition_metadata"] = global_thresholds["_partition_metadata"]
        print("\nPartition Metadata (for WandB logging)")
        print(f"Beta: {meta['partition_metadata']['beta']}")
        print(f"Label Column: {meta['partition_metadata']['label_column']}")
        print(f"IID Type: {meta['partition_metadata']['iid_type']}")
        if "heterogeneity_metrics" in meta["partition_metadata"]:
            print("Heterogeneity Metrics:")
            for k, v in meta["partition_metadata"]["heterogeneity_metrics"].items():
                print(f"  {k}: {v:.4f}")
    
    typer.echo(json.dumps({"status": "ok", "meta": meta}, indent=2))

def run():
    app()

if __name__ == "__main__":
    run()