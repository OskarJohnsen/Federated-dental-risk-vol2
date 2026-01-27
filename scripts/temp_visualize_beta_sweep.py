"""
Temporary script to visualize beta sweep results.

Creates plots with beta values on x-axis and metrics on y-axis, with three curves
(one per paradigm: centralized, local, federated).

Plots:
- F1 Macro (per-client and global)
- Accuracy (per-client and global)
- MSE
- Fleiss Kappa (per-client and global)

Usage:
    python scripts/temp_visualize_beta_sweep.py [--csv-path path/to/wandb_export_beta_runs.csv]
"""

import argparse
import sys
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from digit_fr.core.paths import root_path
from digit_fr.ml.constants import RISK_NAMES

def extract_beta_from_name(run_name: str) -> float:
    """Extract beta value from run name like '...betaL0.5_betaQ0.5...'"""
    match = re.search(r'betaL([0-9.]+)_betaQ([0-9.]+)', run_name)
    if match:
        betaL = float(match.group(1))
        betaQ = float(match.group(2))
        # Only return if betaL == betaQ
        if abs(betaL - betaQ) < 1e-6:
            return betaL
    return None

def find_metric_values(df, pattern):
    """
    Find metric values matching a pattern in the dataframe.
    
    This matches the logic from visualize_results.ipynb exactly.
    Returns the first matching value (for single-run dataframes).
    """
    # Generate all possible pattern variations (matching visualize_results.ipynb exactly)
    patterns_to_try = [
        pattern,  # Original pattern
        f'summary_{pattern}',  # With summary_ prefix
        f'summary_{pattern.replace("test/", "")}',  # Remove test/ then add summary_
        f'history_{pattern}',  # History prefix
        pattern.replace('test/', ''),  # Without test/ prefix
    ]
    
    matching_cols = []
    for pat in patterns_to_try:
        # Substring match (matching visualize_results.ipynb)
        cols = [col for col in df.columns if pat in col]
        matching_cols.extend(cols)
    
    matching_cols = list(dict.fromkeys(matching_cols))
    
    values = []
    for col in matching_cols:
        vals = df[col].dropna().tolist()
        numeric_vals = [v for v in vals if isinstance(v, (int, float)) and not pd.isna(v)]
        values.extend(numeric_vals)
    
    # Return first value (should be single value per run)
    return values[0] if values else None

def get_macro_metric(df, pattern_template, risk_names):
    """Get macro-averaged metric across all risks."""
    values = []
    for risk in risk_names:
        pattern = pattern_template.format(risk=risk)
        val = find_metric_values(df, pattern)
        if val is not None:
            values.append(val)
    return np.mean(values) if values else None

def get_macro_metric_for_beta(paradigm_df, beta_val, pattern_template, debug=False):
    """Get macro-averaged metric for a specific beta value."""
    # Convert beta to float for comparison
    beta_val = float(beta_val)
    beta_df = paradigm_df[abs(paradigm_df['beta'].astype(float) - beta_val) < 1e-6]
    if len(beta_df) == 0:
        if debug:
            print(f"  No runs found for beta={beta_val}")
        return None
    result = get_macro_metric(beta_df, pattern_template, RISK_NAMES)
    if debug and result is None:
        # Debug: show what columns we're searching in
        sample_cols = [c for c in beta_df.columns if any(x in c.lower() for x in ['mse'])]
        print(f"  Pattern '{pattern_template}' not found. Available MSE columns: {sample_cols[:5]}")
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Visualize beta sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to CSV file (defaults to data/results/A/non-iid/sweep_beta/wandb_export_beta_runs.csv)"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    if args.csv_path is None:
        csv_path = root_path('data', 'results', 'A', 'non-iid', 'sweep_beta', 'wandb_export_beta_runs.csv')
    else:
        csv_path = args.csv_path
    
    output_dir = root_path('data', 'results', 'A', 'non-iid', 'sweep_beta')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} runs")
    
    # Extract beta values from run_name, group, or experiment_id
    def extract_beta_from_row(row):
        """Extract beta from run_name, group, or experiment_id."""
        # Try run_name first
        beta = extract_beta_from_name(row.get('run_name', ''))
        if beta is not None:
            return beta
        
        # Try group
        beta = extract_beta_from_name(row.get('group', ''))
        if beta is not None:
            return beta
        
        # Try experiment_id from config
        for col in df.columns:
            if col.startswith('config_experiment_id'):
                beta = extract_beta_from_name(str(row.get(col, '')))
                if beta is not None:
                    return beta
        
        return None
    
    df['beta'] = df.apply(extract_beta_from_row, axis=1)
    df = df[df['beta'].notna()].copy()
    
    if len(df) == 0:
        print("Error: No runs with valid beta values found", file=sys.stderr)
        print("Available columns:", df.columns.tolist()[:20])
        sys.exit(1)
    
    print(f"Found {len(df)} runs with valid beta values")
    
    # Separate by paradigm
    df['paradigm'] = df['group'].apply(lambda x: 'centralized' if 'centralized' in str(x).lower() 
                                       else 'local' if 'local' in str(x).lower()
                                       else 'federated' if 'federated' in str(x).lower()
                                       else None)
    df = df[df['paradigm'].notna()].copy()
    
    print(f"Paradigms: {df['paradigm'].value_counts().to_dict()}")
    
    # Debug: Print sample metric columns to understand naming
    metric_cols = [c for c in df.columns if any(x in c.lower() for x in ['category', 'mse', 'fleiss', 'accuracy', 'f1'])]
    print(f"\nSample metric columns ({len(metric_cols)} total):")
    for col in sorted(metric_cols)[:30]:
        non_null_count = df[col].notna().sum()
        if non_null_count > 0:
            sample_val = df[col].dropna().iloc[0] if non_null_count > 0 else None
            print(f"  {col}: {non_null_count} non-null (sample: {sample_val})")
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    
    beta_values = sorted([float(b) for b in df['beta'].unique()])
    print(f"Beta values: {beta_values}")
    
    # Helper function to get metric for a paradigm and beta
    def get_metric_for_beta(paradigm_df, beta_val, metric_pattern, debug=False):
        """Get metric value for a specific beta value."""
        # Convert beta to float for comparison
        beta_val = float(beta_val)
        beta_df = paradigm_df[abs(paradigm_df['beta'].astype(float) - beta_val) < 1e-6]
        if len(beta_df) == 0:
            if debug:
                print(f"  No runs found for beta={beta_val}")
            return None
        result = find_metric_values(beta_df, metric_pattern)
        if debug and result is None:
            # Debug: show what columns we're searching in
            sample_cols = [c for c in beta_df.columns if any(x in c.lower() for x in ['category', 'mse', 'fleiss'])]
            print(f"  Pattern '{metric_pattern}' not found. Available metric columns: {sample_cols[:5]}")
        return result
    
    # 1. F1 Macro - Per-Client Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_vals = []
        local_vals = []
        fed_vals = []
        
        for beta in beta_values:
            cent_df = df[df['paradigm'] == 'centralized']
            local_df = df[df['paradigm'] == 'local']
            fed_df = df[df['paradigm'] == 'federated']
            
            # Centralized: test/category_per_client_f1_macro_risk_{risk} (macro-averaged)
            # Local: category_per_client_f1_macro_risk_{risk} (no prefix, macro-averaged)  
            # Federated: test/category_per_client_f1_macro_risk_{risk} (macro-averaged)
            
            # Debug first iteration
            debug_this = (idx == 0 and beta == beta_values[0])
            if debug_this:
                print(f"\nDebug - Searching for F1 per-client for {risk}:")
                print(f"  Centralized pattern: test/category_per_client_f1_macro_risk_{risk}")
                print(f"  Local pattern: category_per_client_f1_macro_risk_{risk}")
                print(f"  Federated pattern: test/category_per_client_f1_macro_risk_{risk}")
            
            cent_val = get_metric_for_beta(cent_df, beta, f'test/category_per_client_f1_macro_risk_{risk}', debug=debug_this)
            local_val = get_metric_for_beta(local_df, beta, f'category_per_client_f1_macro_risk_{risk}', debug=debug_this)
            fed_val = get_metric_for_beta(fed_df, beta, f'test/category_per_client_f1_macro_risk_{risk}', debug=debug_this)
            
            cent_vals.append(cent_val if cent_val is not None else np.nan)
            local_vals.append(local_val if local_val is not None else np.nan)
            fed_vals.append(fed_val if fed_val is not None else np.nan)
        
        # Check if we have any valid data
        has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
        
        if has_data:
            if any(not np.isnan(v) for v in cent_vals):
                ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in local_vals):
                ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in fed_vals):
                ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
            
            ax.set_title(f'F1-Macro: {risk}\n(Per-Client Categories)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('F1-Macro Score')
            ax.set_ylim([0, 1])
            if len([v for v in beta_values if v > 0]) > 0:  # Only set log scale if we have positive beta values
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'F1-Macro: {risk}\n(Per-Client Categories)', fontsize=11)
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('F1-Macro Score')
    
    plt.suptitle('F1-Macro vs Beta: Per-Client Categories', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_per_client_beta_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'f1_per_client_beta_sweep.pdf'}")
    
    # 2. F1 Macro - Global Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_vals = []
        local_vals = []
        fed_vals = []
        
        for beta in beta_values:
            cent_df = df[df['paradigm'] == 'centralized']
            local_df = df[df['paradigm'] == 'local']
            fed_df = df[df['paradigm'] == 'federated']
            
            cent_val = get_metric_for_beta(cent_df, beta, f'test/category_global_f1_macro_risk_{risk}')
            local_val = get_metric_for_beta(local_df, beta, f'category_global_f1_macro_risk_{risk}')
            fed_val = get_metric_for_beta(fed_df, beta, f'test/category_global_f1_macro_risk_{risk}')
            
            cent_vals.append(cent_val if cent_val is not None else np.nan)
            local_vals.append(local_val if local_val is not None else np.nan)
            fed_vals.append(fed_val if fed_val is not None else np.nan)
        
        # Check if we have any valid data
        has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
        
        if has_data:
            if any(not np.isnan(v) for v in cent_vals):
                ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in local_vals):
                ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in fed_vals):
                ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
            
            ax.set_title(f'F1-Macro: {risk}\n(Global Categories)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('F1-Macro Score')
            ax.set_ylim([0, 1])
            if len([v for v in beta_values if v > 0]) > 0:
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'F1-Macro: {risk}\n(Global Categories)', fontsize=11)
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('F1-Macro Score')
    
    plt.suptitle('F1-Macro vs Beta: Global Categories', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_global_beta_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'f1_global_beta_sweep.pdf'}")
    
    # 3. Accuracy - Per-Client Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_vals = []
        local_vals = []
        fed_vals = []
        
        for beta in beta_values:
            cent_df = df[df['paradigm'] == 'centralized']
            local_df = df[df['paradigm'] == 'local']
            fed_df = df[df['paradigm'] == 'federated']
            
            cent_val = get_metric_for_beta(cent_df, beta, f'test/category_per_client_accuracy_risk_{risk}')
            local_val = get_metric_for_beta(local_df, beta, f'category_per_client_accuracy_risk_{risk}')
            fed_val = get_metric_for_beta(fed_df, beta, f'test/category_per_client_accuracy_risk_{risk}')
            
            cent_vals.append(cent_val if cent_val is not None else np.nan)
            local_vals.append(local_val if local_val is not None else np.nan)
            fed_vals.append(fed_val if fed_val is not None else np.nan)
        
        # Check if we have any valid data
        has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
        
        if has_data:
            if any(not np.isnan(v) for v in cent_vals):
                ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in local_vals):
                ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in fed_vals):
                ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
            
            ax.set_title(f'Accuracy: {risk}\n(Per-Client Categories)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1])
            if len([v for v in beta_values if v > 0]) > 0:
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Accuracy: {risk}\n(Per-Client Categories)', fontsize=11)
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('Accuracy')
    
    plt.suptitle('Accuracy vs Beta: Per-Client Categories', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_per_client_beta_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_per_client_beta_sweep.pdf'}")
    
    # 4. Accuracy - Global Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_vals = []
        local_vals = []
        fed_vals = []
        
        for beta in beta_values:
            cent_df = df[df['paradigm'] == 'centralized']
            local_df = df[df['paradigm'] == 'local']
            fed_df = df[df['paradigm'] == 'federated']
            
            cent_val = get_metric_for_beta(cent_df, beta, f'test/category_global_accuracy_risk_{risk}')
            local_val = get_metric_for_beta(local_df, beta, f'category_global_accuracy_risk_{risk}')
            fed_val = get_metric_for_beta(fed_df, beta, f'test/category_global_accuracy_risk_{risk}')
            
            cent_vals.append(cent_val if cent_val is not None else np.nan)
            local_vals.append(local_val if local_val is not None else np.nan)
            fed_vals.append(fed_val if fed_val is not None else np.nan)
        
        # Check if we have any valid data
        has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
        
        if has_data:
            if any(not np.isnan(v) for v in cent_vals):
                ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in local_vals):
                ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in fed_vals):
                ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
            
            ax.set_title(f'Accuracy: {risk}\n(Global Categories)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1])
            if len([v for v in beta_values if v > 0]) > 0:
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Accuracy: {risk}\n(Global Categories)', fontsize=11)
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('Accuracy')
    
    plt.suptitle('Accuracy vs Beta: Global Categories', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_global_beta_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_global_beta_sweep.pdf'}")
    
    # 5. MSE (macro-averaged)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    cent_vals = []
    local_vals = []
    fed_vals = []
    
    for beta in beta_values:
        cent_df = df[df['paradigm'] == 'centralized']
        local_df = df[df['paradigm'] == 'local']
        fed_df = df[df['paradigm'] == 'federated']
        
        # MSE is logged per risk, need to calculate macro average
        # Centralized/Federated: test/mse_risk_{risk}
        # Local: /test/mse_risk_{risk} (note leading slash)
        cent_val = get_macro_metric_for_beta(cent_df, beta, 'test/mse_risk_{risk}')
        local_val = get_macro_metric_for_beta(local_df, beta, '/test/mse_risk_{risk}')
        fed_val = get_macro_metric_for_beta(fed_df, beta, 'test/mse_risk_{risk}')
        
        cent_vals.append(cent_val if cent_val is not None else np.nan)
        local_vals.append(local_val if local_val is not None else np.nan)
        fed_vals.append(fed_val if fed_val is not None else np.nan)
    
    # Check if we have any valid data
    has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
    
    if has_data:
        if any(not np.isnan(v) for v in cent_vals):
            ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in local_vals):
            ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in fed_vals):
            ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
        
        ax.set_title('MSE (Macro-Averaged) vs Beta', fontsize=12, fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('MSE')
        if len([v for v in beta_values if v > 0]) > 0:
            ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('MSE (Macro-Averaged) vs Beta', fontsize=12)
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('MSE')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mse_beta_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'mse_beta_sweep.pdf'}")
    
    # 6. Fleiss Kappa - Per-Client Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_vals = []
        local_vals = []
        fed_vals = []
        
        for beta in beta_values:
            cent_df = df[df['paradigm'] == 'centralized']
            local_df = df[df['paradigm'] == 'local']
            fed_df = df[df['paradigm'] == 'federated']
            
            cent_val = get_metric_for_beta(cent_df, beta, f'consistency_per_client/fleiss_kappa_risk_{risk}')
            local_val = get_metric_for_beta(local_df, beta, f'consistency_per_client/fleiss_kappa_risk_{risk}')
            fed_val = get_metric_for_beta(fed_df, beta, f'consistency_per_client/fleiss_kappa_risk_{risk}')
            
            cent_vals.append(cent_val if cent_val is not None else np.nan)
            local_vals.append(local_val if local_val is not None else np.nan)
            fed_vals.append(fed_val if fed_val is not None else np.nan)
        
        # Check if we have any valid data
        has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
        
        if has_data:
            if any(not np.isnan(v) for v in cent_vals):
                ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in local_vals):
                ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in fed_vals):
                ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
            
            ax.set_title(f'Fleiss Kappa: {risk}\n(Per-Client Categories)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('Fleiss Kappa')
            if len([v for v in beta_values if v > 0]) > 0:
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Fleiss Kappa: {risk}\n(Per-Client Categories)', fontsize=11)
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('Fleiss Kappa')
    
    plt.suptitle('Fleiss Kappa vs Beta: Per-Client Categories', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fleiss_kappa_per_client_beta_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fleiss_kappa_per_client_beta_sweep.pdf'}")
    
    # 7. Fleiss Kappa - Global Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_vals = []
        local_vals = []
        fed_vals = []
        
        for beta in beta_values:
            cent_df = df[df['paradigm'] == 'centralized']
            local_df = df[df['paradigm'] == 'local']
            fed_df = df[df['paradigm'] == 'federated']
            
            cent_val = get_metric_for_beta(cent_df, beta, f'consistency_global/fleiss_kappa_risk_{risk}')
            local_val = get_metric_for_beta(local_df, beta, f'consistency_global/fleiss_kappa_risk_{risk}')
            fed_val = get_metric_for_beta(fed_df, beta, f'consistency_global/fleiss_kappa_risk_{risk}')
            
            cent_vals.append(cent_val if cent_val is not None else np.nan)
            local_vals.append(local_val if local_val is not None else np.nan)
            fed_vals.append(fed_val if fed_val is not None else np.nan)
        
        # Check if we have any valid data
        has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
        
        if has_data:
            if any(not np.isnan(v) for v in cent_vals):
                ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in local_vals):
                ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
            if any(not np.isnan(v) for v in fed_vals):
                ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
            
            ax.set_title(f'Fleiss Kappa: {risk}\n(Global Categories)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('Fleiss Kappa')
            if len([v for v in beta_values if v > 0]) > 0:
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Fleiss Kappa: {risk}\n(Global Categories)', fontsize=11)
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('Fleiss Kappa')
    
    plt.suptitle('Fleiss Kappa vs Beta: Global Categories', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fleiss_kappa_global_beta_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fleiss_kappa_global_beta_sweep.pdf'}")
    
    # 8. Summary (Macro-averaged metrics vs Beta)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Note: get_macro_metric_for_beta is defined at module level, not here
    
    # F1-Macro Global Categories
    ax = axes[0, 0]
    cent_vals = []
    local_vals = []
    fed_vals = []
    
    for beta in beta_values:
        cent_df = df[df['paradigm'] == 'centralized']
        local_df = df[df['paradigm'] == 'local']
        fed_df = df[df['paradigm'] == 'federated']
        
        cent_val = get_macro_metric_for_beta(cent_df, beta, 'test/category_global_f1_macro_risk_{risk}')
        local_val = get_macro_metric_for_beta(local_df, beta, 'category_global_f1_macro_risk_{risk}')
        fed_val = get_macro_metric_for_beta(fed_df, beta, 'test/category_global_f1_macro_risk_{risk}')
        
        cent_vals.append(cent_val if cent_val is not None else np.nan)
        local_vals.append(local_val if local_val is not None else np.nan)
        fed_vals.append(fed_val if fed_val is not None else np.nan)
    
    # Check if we have any valid data
    has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
    
    if has_data:
        if any(not np.isnan(v) for v in cent_vals):
            ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in local_vals):
            ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in fed_vals):
            ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
        ax.set_title('F1-Macro (Global Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('F1-Macro Score')
        ax.set_ylim([0, 1])
        if len([v for v in beta_values if v > 0]) > 0:
            ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('F1-Macro (Global Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('F1-Macro Score')
    
    # Accuracy Global Categories
    ax = axes[0, 1]
    cent_vals = []
    local_vals = []
    fed_vals = []
    
    for beta in beta_values:
        cent_df = df[df['paradigm'] == 'centralized']
        local_df = df[df['paradigm'] == 'local']
        fed_df = df[df['paradigm'] == 'federated']
        
        cent_val = get_macro_metric_for_beta(cent_df, beta, 'test/category_global_accuracy_risk_{risk}')
        local_val = get_macro_metric_for_beta(local_df, beta, 'category_global_accuracy_risk_{risk}')
        fed_val = get_macro_metric_for_beta(fed_df, beta, 'test/category_global_accuracy_risk_{risk}')
        
        cent_vals.append(cent_val if cent_val is not None else np.nan)
        local_vals.append(local_val if local_val is not None else np.nan)
        fed_vals.append(fed_val if fed_val is not None else np.nan)
    
    # Check if we have any valid data
    has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
    
    if has_data:
        if any(not np.isnan(v) for v in cent_vals):
            ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in local_vals):
            ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in fed_vals):
            ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
        ax.set_title('Accuracy (Global Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        if len([v for v in beta_values if v > 0]) > 0:
            ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Accuracy (Global Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('Accuracy')
    
    # F1-Macro Per-Client Categories
    ax = axes[0, 2]
    cent_vals = []
    local_vals = []
    fed_vals = []
    
    for beta in beta_values:
        cent_df = df[df['paradigm'] == 'centralized']
        local_df = df[df['paradigm'] == 'local']
        fed_df = df[df['paradigm'] == 'federated']
        
        cent_val = get_macro_metric_for_beta(cent_df, beta, 'test/category_per_client_f1_macro_risk_{risk}')
        local_val = get_macro_metric_for_beta(local_df, beta, 'category_per_client_f1_macro_risk_{risk}')
        fed_val = get_macro_metric_for_beta(fed_df, beta, 'test/category_per_client_f1_macro_risk_{risk}')
        
        cent_vals.append(cent_val if cent_val is not None else np.nan)
        local_vals.append(local_val if local_val is not None else np.nan)
        fed_vals.append(fed_val if fed_val is not None else np.nan)
    
    # Check if we have any valid data
    has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
    
    if has_data:
        if any(not np.isnan(v) for v in cent_vals):
            ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in local_vals):
            ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in fed_vals):
            ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
        ax.set_title('F1-Macro (Per-Client Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('F1-Macro Score')
        ax.set_ylim([0, 1])
        if len([v for v in beta_values if v > 0]) > 0:
            ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('F1-Macro (Per-Client Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('F1-Macro Score')
    
    # Accuracy Per-Client Categories
    ax = axes[1, 0]
    cent_vals = []
    local_vals = []
    fed_vals = []
    
    for beta in beta_values:
        cent_df = df[df['paradigm'] == 'centralized']
        local_df = df[df['paradigm'] == 'local']
        fed_df = df[df['paradigm'] == 'federated']
        
        cent_val = get_macro_metric_for_beta(cent_df, beta, 'test/category_per_client_accuracy_risk_{risk}')
        local_val = get_macro_metric_for_beta(local_df, beta, 'category_per_client_accuracy_risk_{risk}')
        fed_val = get_macro_metric_for_beta(fed_df, beta, 'test/category_per_client_accuracy_risk_{risk}')
        
        cent_vals.append(cent_val if cent_val is not None else np.nan)
        local_vals.append(local_val if local_val is not None else np.nan)
        fed_vals.append(fed_val if fed_val is not None else np.nan)
    
    # Check if we have any valid data
    has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
    
    if has_data:
        if any(not np.isnan(v) for v in cent_vals):
            ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in local_vals):
            ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in fed_vals):
            ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
        ax.set_title('Accuracy (Per-Client Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        if len([v for v in beta_values if v > 0]) > 0:
            ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Accuracy (Per-Client Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('Accuracy')
    
    # MSE (Macro-averaged)
    ax = axes[1, 1]
    cent_vals = []
    local_vals = []
    fed_vals = []
    
    for beta in beta_values:
        cent_df = df[df['paradigm'] == 'centralized']
        local_df = df[df['paradigm'] == 'local']
        fed_df = df[df['paradigm'] == 'federated']
        
        # MSE is logged per risk, need to calculate macro average
        # Centralized/Federated: test/mse_risk_{risk}
        # Local: /test/mse_risk_{risk} (note leading slash)
        cent_val = get_macro_metric_for_beta(cent_df, beta, 'test/mse_risk_{risk}')
        local_val = get_macro_metric_for_beta(local_df, beta, '/test/mse_risk_{risk}')
        fed_val = get_macro_metric_for_beta(fed_df, beta, 'test/mse_risk_{risk}')
        
        cent_vals.append(cent_val if cent_val is not None else np.nan)
        local_vals.append(local_val if local_val is not None else np.nan)
        fed_vals.append(fed_val if fed_val is not None else np.nan)
    
    # Check if we have any valid data
    has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
    
    if has_data:
        if any(not np.isnan(v) for v in cent_vals):
            ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in local_vals):
            ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in fed_vals):
            ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
        ax.set_title('MSE (Probability Metrics)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('MSE')
        if len([v for v in beta_values if v > 0]) > 0:
            ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('MSE (Probability Metrics)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('MSE')
    
    # Fleiss Kappa Per-Client
    ax = axes[1, 2]
    cent_vals = []
    local_vals = []
    fed_vals = []
    
    for beta in beta_values:
        cent_df = df[df['paradigm'] == 'centralized']
        local_df = df[df['paradigm'] == 'local']
        fed_df = df[df['paradigm'] == 'federated']
        
        cent_val = get_macro_metric_for_beta(cent_df, beta, 'consistency_per_client/fleiss_kappa_risk_{risk}')
        local_val = get_macro_metric_for_beta(local_df, beta, 'consistency_per_client/fleiss_kappa_risk_{risk}')
        fed_val = get_macro_metric_for_beta(fed_df, beta, 'consistency_per_client/fleiss_kappa_risk_{risk}')
        
        cent_vals.append(cent_val if cent_val is not None else np.nan)
        local_vals.append(local_val if local_val is not None else np.nan)
        fed_vals.append(fed_val if fed_val is not None else np.nan)
    
    # Check if we have any valid data
    has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
    
    if has_data:
        if any(not np.isnan(v) for v in cent_vals):
            ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in local_vals):
            ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in fed_vals):
            ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
        ax.set_title('Fleiss Kappa (Per-Client Thresholds)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('Fleiss Kappa')
        if len([v for v in beta_values if v > 0]) > 0:
            ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Fleiss Kappa (Per-Client Thresholds)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('Fleiss Kappa')
    
    # Fleiss Kappa Global
    ax = axes[2, 0]
    cent_vals = []
    local_vals = []
    fed_vals = []
    
    for beta in beta_values:
        cent_df = df[df['paradigm'] == 'centralized']
        local_df = df[df['paradigm'] == 'local']
        fed_df = df[df['paradigm'] == 'federated']
        
        cent_val = get_macro_metric_for_beta(cent_df, beta, 'consistency_global/fleiss_kappa_risk_{risk}')
        local_val = get_macro_metric_for_beta(local_df, beta, 'consistency_global/fleiss_kappa_risk_{risk}')
        fed_val = get_macro_metric_for_beta(fed_df, beta, 'consistency_global/fleiss_kappa_risk_{risk}')
        
        cent_vals.append(cent_val if cent_val is not None else np.nan)
        local_vals.append(local_val if local_val is not None else np.nan)
        fed_vals.append(fed_val if fed_val is not None else np.nan)
    
    # Check if we have any valid data
    has_data = any(not np.isnan(v) for v in cent_vals + local_vals + fed_vals)
    
    if has_data:
        if any(not np.isnan(v) for v in cent_vals):
            ax.plot(beta_values, cent_vals, 'o-', label='Centralized', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in local_vals):
            ax.plot(beta_values, local_vals, 's-', label='Local', linewidth=2, markersize=8)
        if any(not np.isnan(v) for v in fed_vals):
            ax.plot(beta_values, fed_vals, '^-', label='Federated', linewidth=2, markersize=8)
        ax.set_title('Fleiss Kappa (Global Thresholds)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('Fleiss Kappa')
        if len([v for v in beta_values if v > 0]) > 0:
            ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Fleiss Kappa (Global Thresholds)\nMacro-Averaged', fontweight='bold')
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('Fleiss Kappa')
    
    # Empty subplots for alignment
    ax = axes[2, 1]
    ax.axis('off')
    ax = axes[2, 2]
    ax.axis('off')
    
    plt.suptitle('Macro-Averaged Metrics Summary vs Beta\n(Centralized vs Local vs Federated Learning)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_beta_sweep.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'summary_beta_sweep.pdf'}")
    
    print("\nAll visualizations saved!")

if __name__ == "__main__":
    main()
