"""
Export WandB run summary/results as CSV.

This script extracts metrics from a specific WandB run and saves them to CSV.
The output is saved to data/results/{DATASET}/{IID_TYPE}/wandb_export_all_metrics.csv

Usage:
    python scripts/export_wandb_run.py <run_id>
    python scripts/export_wandb_run.py <run_id> --project my-project --entity my-entity
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import wandb
from digit_fr.core.paths import ensure_dir, root_path
from digit_fr.ml.constants import DATASET, IID_TYPE
from digit_fr.ml.util.wandb_config import get_wandb_project, get_wandb_entity

def export_run_to_csv(run_id: str, project: str = None, entity: str = None, output_path: Path = None) -> Path:
    """
    Export a single WandB run to CSV.
    
    Args:
        run_id: WandB run ID
        project: WandB project name (defaults to get_wandb_project())
        entity: WandB entity name (defaults to get_wandb_entity())
        output_path: Path to save CSV (defaults to data/results/{DATASET}/{IID_TYPE}/wandb_export_all_metrics.csv)
    
    Returns:
        Path to saved CSV file
    """
    if project is None:
        project = get_wandb_project()
    if entity is None:
        entity = get_wandb_entity()
    
    if output_path is None:
        output_path = root_path('data', 'results', DATASET, IID_TYPE, 'wandb_export_all_metrics.csv')
    
    ensure_dir(output_path.parent)
    
    api = wandb.Api(timeout=60)
    
    # Construct project path
    if entity:
        project_path = f"{entity}/{project}"
    else:
        project_path = project
    
    print(f"Fetching run {run_id} from project: {project_path}")
    
    try:
        run = api.run(f"{project_path}/{run_id}")
    except Exception as e:
        print(f"Error: Could not fetch run {run_id}: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing run: {run.name} (ID: {run_id})")
    
    # Extract run metadata
    created_at = None
    created_at_val = getattr(run, 'created_at', None)
    if created_at_val:
        if hasattr(created_at_val, 'isoformat'):
            created_at = created_at_val.isoformat()
        else:
            created_at = str(created_at_val)
    
    updated_at = None
    updated_at_val = getattr(run, 'updated_at', None)
    if updated_at_val:
        if hasattr(updated_at_val, 'isoformat'):
            updated_at = updated_at_val.isoformat()
        else:
            updated_at = str(updated_at_val)
    
    run_tags = getattr(run, 'tags', None)
    tags_str = ", ".join(run_tags) if run_tags else ""
    
    record = {
        "run_id": run_id,
        "run_name": run.name or getattr(run, 'name', ''),
        "group": getattr(run, 'group', ''),
        "job_type": getattr(run, 'job_type', None),
        "state": getattr(run, 'state', None),
        "tags": tags_str,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    
    # Extract config
    if run.config:
        if isinstance(run.config, dict):
            config_dict = run.config
        elif hasattr(run.config, '_items'):
            config_dict = dict(run.config._items)
        else:
            config_dict = {}
        
        for key, value in config_dict.items():
            if isinstance(value, (dict, list)):
                record[f"config_{key}"] = json.dumps(value)
            else:
                record[f"config_{key}"] = value
    else:
        config_dict = {}
    
    # Extract summary metrics
    try:
        summary_dict = {}
        if run.summary:
            if isinstance(run.summary, dict):
                summary_dict = run.summary
            elif hasattr(run.summary, '_json_dict') and hasattr(run.summary._json_dict, 'keys'):
                try:
                    summary_dict = dict(run.summary._json_dict)
                except:
                    summary_dict = {}
            elif hasattr(run.summary, 'items'):
                try:
                    summary_dict = dict(run.summary.items())
                except:
                    summary_dict = {}
            elif isinstance(run.summary, str):
                print(f"Warning: Summary for run {run_id} is a string, skipping summary extraction")
                summary_dict = {}
            else:
                print(f"Warning: Summary for run {run_id} is not a dict-like object (type: {type(run.summary)})")
                summary_dict = {}
        
        for key, value in summary_dict.items():
            if key.startswith('_'):
                continue
            
            if isinstance(value, (dict, list)):
                record[f"summary_{key}"] = json.dumps(value)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                record[f"summary_{key}"] = value
            else:
                try:
                    record[f"summary_{key}"] = str(value)
                except:
                    record[f"summary_{key}"] = None
    
    except Exception as e:
        print(f"Warning: Could not process summary for run {run_id}: {e}")
    
    # Create DataFrame and save
    df = pd.DataFrame([record])
    
    print(f"\nExported 1 run with {len(df.columns)} columns")
    print(f"Columns: {sorted(df.columns)}")
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export WandB run summary/results as CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "run_id",
        help="WandB run ID to export"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help=f"WandB project name (defaults to {get_wandb_project()})"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity name (defaults to logged-in user)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output CSV path (defaults to data/results/{DATASET}/{IID_TYPE}/wandb_export_all_metrics.csv)"
    )
    
    args = parser.parse_args()
    
    export_run_to_csv(
        run_id=args.run_id,
        project=args.project,
        entity=args.entity,
        output_path=args.output
    )

if __name__ == "__main__":
    main()