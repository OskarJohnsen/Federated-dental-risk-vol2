from __future__ import annotations
from pathlib import Path
import json
import typer
import numpy as np
from ..config.loader import load_all_configs
from ..generation.synth import generate_dataset
from ...core.paths import ensure_dir, root_path
from digit_fr.ml.constants import DATASET, IID_TYPE

app = typer.Typer(add_completion=False)

@app.command()
def main(
    seed: int | None = typer.Option(None, help="Random seed override"),
    output_dir: Path | None = typer.Option(None, help="Output directory; defaults to config output.output_dir from project root"),
    formats: str = typer.Option("csv,xlsx", help="Comma-separated formats: csv,xlsx"),
):
    """Generate the synthetic dataset"""
    configs = load_all_configs()
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(configs["generation"]["dataset"]["random_seed"])

    df, global_thresholds = generate_dataset(configs)

    cfg_out_dir = configs["generation"]["output"]["output_dir"]
    base = configs["generation"]["output"]["filename_base"]
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
    print(f"threshold path: {thresholds_path}")
    with thresholds_path.open("w") as f:
        json.dump(global_thresholds, f, indent=2)

    meta = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "formats": fmts,
        "output_dir": str(out_dir.resolve()),
    }
    typer.echo(json.dumps({"status": "ok", "meta": meta}, indent=2))

def run():
    app()

if __name__ == "__main__":
    run()