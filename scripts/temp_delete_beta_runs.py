"""
Temporary script to delete WandB runs that have "beta" in their name.

Usage:
    python scripts/temp_delete_beta_runs.py [--dry-run]
    python scripts/temp_delete_beta_runs.py --project my-project --entity my-entity
"""

import argparse
import sys
import wandb
from digit_fr.ml.util.wandb_config import get_wandb_project, get_wandb_entity

def delete_beta_runs(project: str = None, entity: str = None, dry_run: bool = True):
    """
    Delete all WandB runs that have "beta" in their name.
    
    Args:
        project: WandB project name (defaults to get_wandb_project())
        entity: WandB entity name (defaults to get_wandb_entity())
        dry_run: If True, only print what would be deleted without actually deleting
    """
    if project is None:
        project = get_wandb_project()
    if entity is None:
        entity = get_wandb_entity()
    
    api = wandb.Api(timeout=60)
    
    # Construct project path
    if entity:
        project_path = f"{entity}/{project}"
    else:
        project_path = project
    
    print(f"Searching for runs with 'beta' in name in project: {project_path}")
    if dry_run:
        print("DRY RUN MODE - No runs will be deleted")
    else:
        print("LIVE MODE - Runs will be permanently deleted!")
    
    try:
        runs = api.runs(project_path)
        beta_runs = []
        
        for run in runs:
            run_name = (run.name or "").lower()
            group_name = (getattr(run, 'group', '') or "").lower()
            experiment_id = ""
            if run.config:
                if isinstance(run.config, dict):
                    experiment_id = (run.config.get('experiment_id', '') or "").lower()
                elif hasattr(run.config, '_items'):
                    try:
                        config_dict = dict(run.config._items)
                        experiment_id = (config_dict.get('experiment_id', '') or "").lower()
                    except:
                        pass
            
            # Check if "beta" appears in name, group, or experiment_id
            if "beta" in run_name or "beta" in group_name or "beta" in experiment_id:
                beta_runs.append((run.id, run.name or "", getattr(run, 'group', ''), run.state))
        
        if not beta_runs:
            print("No runs with 'beta' in name found.")
            return
        
        print(f"\nFound {len(beta_runs)} runs with 'beta' in name/group/experiment_id:")
        for run_id, run_name, group_name, state in beta_runs:
            print(f"  - {run_id}: {run_name} (group: {group_name}, state: {state})")
        
        if dry_run:
            print(f"\nDRY RUN: Would delete {len(beta_runs)} runs")
            print("Run with --no-dry-run to actually delete them")
        else:
            print(f"\nDeleting {len(beta_runs)} runs...")
            deleted = 0
            failed = 0
            
            for run_id, run_name, group_name, state in beta_runs:
                try:
                    run = api.run(f"{project_path}/{run_id}")
                    run.delete()
                    print(f"  ✓ Deleted: {run_id} ({run_name}, group: {group_name})")
                    deleted += 1
                except Exception as e:
                    print(f"  ✗ Failed to delete {run_id}: {e}", file=sys.stderr)
                    failed += 1
            
            print(f"\nSummary: {deleted} deleted, {failed} failed")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Delete WandB runs with 'beta' in their name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode (default: True, no deletion)"
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Actually delete runs (disable dry run)"
    )
    
    args = parser.parse_args()
    
    delete_beta_runs(
        project=args.project,
        entity=args.entity,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
