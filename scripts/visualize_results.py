"""
Visualize WandB results.

reads the CSV exported by export_wandb_run.py and creates
the visualizations, saving them as PDF files todata/results/{DATASET}/{IID_TYPE}/

Usage:
    python scripts/visualize_results.py [--csv-path path/to/wandb_export_all_metrics.csv]

Disclaimer: This file is copied from the internet and adjusted by AI
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fdrp.core.paths import root_path
from fdrp.ml.constants import DATASET, IID_TYPE, RISK_NAMES

def find_metric_values(df, pattern):
    """
    Find metric values matching a pattern in the dataframe.
    
    Tries multiple variations of the pattern to handle different naming conventions.
    """
    patterns_to_try = [
        pattern,
        f'summary_{pattern}',
        f'summary_{pattern.replace("test/", "")}',
        f'history_{pattern}',
        pattern.replace('test/', ''),
    ]
    
    matching_cols = []
    for pat in patterns_to_try:
        cols = [col for col in df.columns if pat in col]
        matching_cols.extend(cols)
    
    matching_cols = list(dict.fromkeys(matching_cols))
    
    values = []
    for col in matching_cols:
        vals = df[col].dropna().tolist()
        numeric_vals = [v for v in vals if isinstance(v, (int, float)) and not pd.isna(v)]
        values.extend(numeric_vals)
    
    return values if values else None

def get_macro_avg(metrics_dict, level, metric_type):
    """Calculate macro-averaged metric across all risks."""
    values = []
    for risk in RISK_NAMES:
        risk_values = metrics_dict.get(level, {}).get(metric_type, {}).get(risk)
        if risk_values:
            values.extend(risk_values)
    return values if values else None

def main():
    parser = argparse.ArgumentParser(
        description="Visualize WandB results exactly as visualize_results.ipynb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help=f"Path to CSV file (defaults to data/results/{DATASET}/{IID_TYPE}/wandb_export_all_metrics.csv)"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    if args.csv_path is None:
        csv_path = root_path('data', 'results', DATASET, IID_TYPE, 'wandb_export_all_metrics.csv')
    else:
        csv_path = args.csv_path
    
    output_dir = root_path('data', 'results', DATASET, IID_TYPE)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} runs")
    print(f"types: {df['group'].value_counts().to_dict()}")
    
    # Separate runs by type
    centralized_runs = df[df['group'].str.contains('centralized', case=False, na=False)].copy()
    local_runs = df[df['group'].str.contains('local', case=False, na=False)].copy()
    federated_runs = df[df['group'].str.contains('federated', case=False, na=False)].copy()
    
    print(f"Centralized runs: {len(centralized_runs)}")
    print(f"Local runs: {len(local_runs)}")
    print(f"Federated runs: {len(federated_runs)}")
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    
    # Extract metrics
    centralized_metrics = {
        'per_client': {
            'f1_macro': {risk: find_metric_values(centralized_runs, f'test/category_per_client_f1_macro_risk_{risk}') 
                         for risk in RISK_NAMES},
            'accuracy': {risk: find_metric_values(centralized_runs, f'test/category_per_client_accuracy_risk_{risk}') 
                         for risk in RISK_NAMES}
        },
        'global': {
            'f1_macro': {risk: find_metric_values(centralized_runs, f'test/category_global_f1_macro_risk_{risk}') 
                         for risk in RISK_NAMES},
            'accuracy': {risk: find_metric_values(centralized_runs, f'test/category_global_accuracy_risk_{risk}') 
                         for risk in RISK_NAMES}
        },
        'probability': {
            'mse': {risk: find_metric_values(centralized_runs, f'test/mse_risk_{risk}') 
                    for risk in RISK_NAMES},
            'ece': {risk: find_metric_values(centralized_runs, f'test/ece_prob_risk_{risk}') 
                    for risk in RISK_NAMES}
        }
    }
    
    local_metrics = {
        'per_client': {
            'f1_macro': {risk: find_metric_values(local_runs, f'category_per_client_f1_macro_risk_{risk}') 
                         for risk in RISK_NAMES},
            'accuracy': {risk: find_metric_values(local_runs, f'category_per_client_accuracy_risk_{risk}') 
                         for risk in RISK_NAMES}
        },
        'global': {
            'f1_macro': {risk: find_metric_values(local_runs, f'category_global_f1_macro_risk_{risk}') 
                         for risk in RISK_NAMES},
            'accuracy': {risk: find_metric_values(local_runs, f'category_global_accuracy_risk_{risk}') 
                         for risk in RISK_NAMES}
        },
        'probability': {
            'mse': {risk: find_metric_values(local_runs, f'/test/mse_risk_{risk}') 
                    for risk in RISK_NAMES},
            'ece': {risk: find_metric_values(local_runs, f'/test/ece_prob_risk_{risk}') 
                    for risk in RISK_NAMES}
        }
    }
    
    centralized_consistency = {
        'per_client': {
            'patient_disagreement': {risk: find_metric_values(centralized_runs, f'consistency_per_client/patient_disagreement_risk_{risk}') 
                                     for risk in RISK_NAMES},
            'fleiss_kappa': {risk: find_metric_values(centralized_runs, f'consistency_per_client/fleiss_kappa_risk_{risk}') 
                             for risk in RISK_NAMES}
        },
        'global': {
            'patient_disagreement': {risk: find_metric_values(centralized_runs, f'consistency_global/patient_disagreement_risk_{risk}') 
                                      for risk in RISK_NAMES},
            'fleiss_kappa': {risk: find_metric_values(centralized_runs, f'consistency_global/fleiss_kappa_risk_{risk}') 
                             for risk in RISK_NAMES}
        }
    }
    
    local_consistency = {
        'per_client': {
            'patient_disagreement': {risk: find_metric_values(local_runs, f'consistency_per_client/patient_disagreement_risk_{risk}') 
                                     for risk in RISK_NAMES},
            'fleiss_kappa': {risk: find_metric_values(local_runs, f'consistency_per_client/fleiss_kappa_risk_{risk}') 
                             for risk in RISK_NAMES}
        },
        'global': {
            'patient_disagreement': {risk: find_metric_values(local_runs, f'consistency_global/patient_disagreement_risk_{risk}') 
                                     for risk in RISK_NAMES},
            'fleiss_kappa': {risk: find_metric_values(local_runs, f'consistency_global/fleiss_kappa_risk_{risk}') 
                             for risk in RISK_NAMES}
        }
    }
    
    federated_metrics = {
        'per_client': {
            'f1_macro': {risk: find_metric_values(federated_runs, f'test/category_per_client_f1_macro_risk_{risk}') 
                         for risk in RISK_NAMES},
            'accuracy': {risk: find_metric_values(federated_runs, f'test/category_per_client_accuracy_risk_{risk}') 
                         for risk in RISK_NAMES}
        },
        'global': {
            'f1_macro': {risk: find_metric_values(federated_runs, f'test/category_global_f1_macro_risk_{risk}') 
                         for risk in RISK_NAMES},
            'accuracy': {risk: find_metric_values(federated_runs, f'test/category_global_accuracy_risk_{risk}') 
                         for risk in RISK_NAMES}
        },
        'probability': {
            'mse': {risk: find_metric_values(federated_runs, f'test/mse_risk_{risk}') 
                    for risk in RISK_NAMES},
            'ece': {risk: find_metric_values(federated_runs, f'test/ece_prob_risk_{risk}') 
                    for risk in RISK_NAMES}
        }
    }
    
    federated_consistency = {
        'per_client': {
            'patient_disagreement': {risk: find_metric_values(federated_runs, f'consistency_per_client/patient_disagreement_risk_{risk}') 
                                     for risk in RISK_NAMES},
            'fleiss_kappa': {risk: find_metric_values(federated_runs, f'consistency_per_client/fleiss_kappa_risk_{risk}') 
                             for risk in RISK_NAMES}
        },
        'global': {
            'patient_disagreement': {risk: find_metric_values(federated_runs, f'consistency_global/patient_disagreement_risk_{risk}') 
                                     for risk in RISK_NAMES},
            'fleiss_kappa': {risk: find_metric_values(federated_runs, f'consistency_global/fleiss_kappa_risk_{risk}') 
                             for risk in RISK_NAMES}
        }
    }
    
    # Create visualizations
    
    # 1. F1-Macro Per-Client Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_f1 = centralized_metrics['per_client']['f1_macro'][risk]
        local_f1 = local_metrics['per_client']['f1_macro'][risk]
        fed_f1 = federated_metrics['per_client']['f1_macro'][risk]
        
        data_to_plot = []
        labels = []
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        if cent_f1:
            data_to_plot.append(cent_f1)
            labels.append('Centralized')
        if local_f1:
            data_to_plot.append(local_f1)
            labels.append('Local')
        if fed_f1:
            data_to_plot.append(fed_f1)
            labels.append('Federated')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.set_title(f'F1-Macro: {risk}\n(Per-Client Categories)', fontsize=11, fontweight='bold')
            ax.set_ylabel('F1-Macro Score')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'F1-Macro: {risk}\n(Per-Client Categories)', fontsize=11)
    
    plt.suptitle(f'F1-Macro {DATASET}_{IID_TYPE}: Per-Client Categories\n(Centralized vs Local vs Federated Learning)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_per_client.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'f1_per_client.pdf'}")
    
    # 2. F1-Macro Global Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_f1 = centralized_metrics['global']['f1_macro'][risk]
        local_f1 = local_metrics['global']['f1_macro'][risk]
        fed_f1 = federated_metrics['global']['f1_macro'][risk]
        
        data_to_plot = []
        labels = []
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        if cent_f1:
            data_to_plot.append(cent_f1)
            labels.append('Centralized')
        if local_f1:
            data_to_plot.append(local_f1)
            labels.append('Local')
        if fed_f1:
            data_to_plot.append(fed_f1)
            labels.append('Federated')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.set_title(f'F1-Macro: {risk}\n(Global Categories)', fontsize=11, fontweight='bold')
            ax.set_ylabel('F1-Macro Score')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'F1-Macro: {risk}\n(Global Categories)', fontsize=11)
    
    plt.suptitle(f'F1-Macro {DATASET}_{IID_TYPE}: Global Categories\n(Centralized vs Local vs Federated Learning)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_global.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'f1_global.pdf'}")
    
    # 3. Accuracy Per-Client Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_acc = centralized_metrics['per_client']['accuracy'][risk]
        local_acc = local_metrics['per_client']['accuracy'][risk]
        fed_acc = federated_metrics['per_client']['accuracy'][risk]
        
        data_to_plot = []
        labels = []
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        if cent_acc:
            data_to_plot.append(cent_acc)
            labels.append('Centralized')
        if local_acc:
            data_to_plot.append(local_acc)
            labels.append('Local')
        if fed_acc:
            data_to_plot.append(fed_acc)
            labels.append('Federated')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.set_title(f'Accuracy: {risk}\n(Per-Client Categories)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Accuracy: {risk}\n(Per-Client Categories)', fontsize=11)
    
    plt.suptitle(f'Accuracy {DATASET}_{IID_TYPE}: Per-Client Categories\n(Centralized vs Local vs Federated Learning)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_per_client.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_per_client.pdf'}")
    
    # 4. Accuracy Global Categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_acc = centralized_metrics['global']['accuracy'][risk]
        local_acc = local_metrics['global']['accuracy'][risk]
        fed_acc = federated_metrics['global']['accuracy'][risk]
        
        data_to_plot = []
        labels = []
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        if cent_acc:
            data_to_plot.append(cent_acc)
            labels.append('Centralized')
        if local_acc:
            data_to_plot.append(local_acc)
            labels.append('Local')
        if fed_acc:
            data_to_plot.append(fed_acc)
            labels.append('Federated')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.set_title(f'Accuracy: {risk}\n(Global Categories)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Accuracy: {risk}\n(Global Categories)', fontsize=11)
    
    plt.suptitle(f'Accuracy {DATASET}_{IID_TYPE}: Global Categories\n(Centralized vs Local vs Federated Learning)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'acc_global.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'acc_global.pdf'}")
    
    # 5. MSE
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_mse = centralized_metrics['probability']['mse'][risk]
        local_mse = local_metrics['probability']['mse'][risk]
        fed_mse = federated_metrics['probability']['mse'][risk]
        
        data_to_plot = []
        labels = []
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        if cent_mse:
            data_to_plot.append(cent_mse)
            labels.append('Centralized')
        if local_mse:
            data_to_plot.append(local_mse)
            labels.append('Local')
        if fed_mse:
            data_to_plot.append(fed_mse)
            labels.append('Federated')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.set_title(f'MSE: {risk}\n(Predicted vs True Probabilities)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean Squared Error')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'MSE: {risk}', fontsize=11)
    
    plt.suptitle(f'MSE {DATASET}_{IID_TYPE}: Predicted vs True Probabilities\n(Centralized vs Local vs Federated Learning)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'mse.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'mse.pdf'}")
    
    # 6. ECE
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_ece = centralized_metrics['probability']['ece'][risk]
        local_ece = local_metrics['probability']['ece'][risk]
        fed_ece = federated_metrics['probability']['ece'][risk]
        
        data_to_plot = []
        labels = []
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        if cent_ece:
            data_to_plot.append(cent_ece)
            labels.append('Centralized')
        if local_ece:
            data_to_plot.append(local_ece)
            labels.append('Local')
        if fed_ece:
            data_to_plot.append(fed_ece)
            labels.append('Federated')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.set_title(f'ECE: {risk}\n(Predicted vs True Probabilities)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Expected Calibration Error')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'ECE: {risk}', fontsize=11)
    
    plt.suptitle(f'ECE {DATASET}_{IID_TYPE}: Predicted vs True Probabilities\n(Centralized vs Local vs Federated Learning)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'ece.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'ece.pdf'}")
    
    # 7. Fleiss Kappa Per-Client
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_fleiss = centralized_consistency['per_client']['fleiss_kappa'][risk]
        local_fleiss = local_consistency['per_client']['fleiss_kappa'][risk]
        fed_fleiss = federated_consistency['per_client']['fleiss_kappa'][risk]
        
        data_to_plot = []
        labels = []
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        if cent_fleiss:
            data_to_plot.append(cent_fleiss)
            labels.append('Centralized')
        if local_fleiss:
            data_to_plot.append(local_fleiss)
            labels.append('Local')
        if fed_fleiss:
            data_to_plot.append(fed_fleiss)
            labels.append('Federated')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.set_title(f'Fleiss Kappa: {risk}\n(Per-Client Thresholds)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Fleiss Kappa')
            ax.set_ylim([-1, 1])
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Fleiss Kappa: {risk}\n(Per-Client Thresholds)', fontsize=11)
    
    plt.suptitle(f'Fleiss Kappa {DATASET}_{IID_TYPE}: Per-Client Thresholds\n(Centralized vs Local vs Federated Learning)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fleiss_kappa_per_client.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fleiss_kappa_per_client.pdf'}")
    
    # 8. Fleiss Kappa Global
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        cent_fleiss = centralized_consistency['global']['fleiss_kappa'][risk]
        local_fleiss = local_consistency['global']['fleiss_kappa'][risk]
        fed_fleiss = federated_consistency['global']['fleiss_kappa'][risk]
        
        data_to_plot = []
        labels = []
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        if cent_fleiss:
            data_to_plot.append(cent_fleiss)
            labels.append('Centralized')
        if local_fleiss:
            data_to_plot.append(local_fleiss)
            labels.append('Local')
        if fed_fleiss:
            data_to_plot.append(fed_fleiss)
            labels.append('Federated')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.set_title(f'Fleiss Kappa: {risk}\n(Global Thresholds)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Fleiss Kappa')
            ax.set_ylim([-1, 1])
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Fleiss Kappa: {risk}\n(Global Thresholds)', fontsize=11)
    
    plt.suptitle(f'Fleiss Kappa {DATASET}_{IID_TYPE}: Global Thresholds\n(Centralized vs Local vs Federated Learning)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fleiss_kappa_global.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fleiss_kappa_global.pdf'}")
    
    # 9. Patient Disagreement
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, risk in enumerate(RISK_NAMES):
        ax = axes[idx]
        
        # Per-client thresholds
        cent_per_client = centralized_consistency['per_client']['patient_disagreement'][risk]
        local_per_client = local_consistency['per_client']['patient_disagreement'][risk]
        fed_per_client = federated_consistency['per_client']['patient_disagreement'][risk]
        
        # Global thresholds
        cent_global = centralized_consistency['global']['patient_disagreement'][risk]
        local_global = local_consistency['global']['patient_disagreement'][risk]
        fed_global = federated_consistency['global']['patient_disagreement'][risk]
        
        data_to_plot = []
        labels = []
        
        if cent_per_client:
            data_to_plot.append(cent_per_client)
            labels.append('Centralized\n(Per-Client)')
        if local_per_client:
            data_to_plot.append(local_per_client)
            labels.append('Local\n(Per-Client)')
        if fed_per_client:
            data_to_plot.append(fed_per_client)
            labels.append('Federated\n(Per-Client)')
        if cent_global:
            data_to_plot.append(cent_global)
            labels.append('Centralized\n(Global)')
        if local_global:
            data_to_plot.append(local_global)
            labels.append('Local\n(Global)')
        if fed_global:
            data_to_plot.append(fed_global)
            labels.append('Federated\n(Global)')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightcyan']
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.set_title(f'Patient Disagreement: {risk}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Patient Disagreement')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Patient Disagreement: {risk}', fontsize=11)
    
    plt.suptitle(f'Patient Disagreement {DATASET}_{IID_TYPE}\n(Centralized vs Local vs Federated Learning)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'patient_disagreement.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'patient_disagreement.pdf'}")
    
    # 10. Summary (Macro-averaged metrics)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # F1-Macro Global
    ax = axes[0, 0]
    cent_f1 = get_macro_avg(centralized_metrics, 'global', 'f1_macro')
    local_f1 = get_macro_avg(local_metrics, 'global', 'f1_macro')
    fed_f1 = get_macro_avg(federated_metrics, 'global', 'f1_macro')
    data_to_plot = [v for v in [cent_f1, local_f1, fed_f1] if v]
    labels = ['Centralized' if cent_f1 else '', 'Local' if local_f1 else '', 'Federated' if fed_f1 else '']
    labels = [l for l in labels if l]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
        ax.set_title('F1-Macro (Global Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_ylabel('F1-Macro Score')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    # Accuracy Global
    ax = axes[0, 1]
    cent_acc = get_macro_avg(centralized_metrics, 'global', 'accuracy')
    local_acc = get_macro_avg(local_metrics, 'global', 'accuracy')
    fed_acc = get_macro_avg(federated_metrics, 'global', 'accuracy')
    data_to_plot = [v for v in [cent_acc, local_acc, fed_acc] if v]
    labels = ['Centralized' if cent_acc else '', 'Local' if local_acc else '', 'Federated' if fed_acc else '']
    labels = [l for l in labels if l]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
        ax.set_title('Accuracy (Global Categories)\nMacro-Averaged', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    # MSE
    ax = axes[1, 0]
    cent_mse = get_macro_avg(centralized_metrics, 'probability', 'mse')
    local_mse = get_macro_avg(local_metrics, 'probability', 'mse')
    fed_mse = get_macro_avg(federated_metrics, 'probability', 'mse')
    data_to_plot = [v for v in [cent_mse, local_mse, fed_mse] if v]
    labels = ['Centralized' if cent_mse else '', 'Local' if local_mse else '', 'Federated' if fed_mse else '']
    labels = [l for l in labels if l]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
        ax.set_title('MSE (Probability Metrics)\nMacro-Averaged', fontweight='bold')
        ax.set_ylabel('MSE')
        ax.grid(True, alpha=0.3)
    
    # ECE
    ax = axes[1, 1]
    cent_ece = get_macro_avg(centralized_metrics, 'probability', 'ece')
    local_ece = get_macro_avg(local_metrics, 'probability', 'ece')
    fed_ece = get_macro_avg(federated_metrics, 'probability', 'ece')
    data_to_plot = [v for v in [cent_ece, local_ece, fed_ece] if v]
    labels = ['Centralized' if cent_ece else '', 'Local' if local_ece else '', 'Federated' if fed_ece else '']
    labels = [l for l in labels if l]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
        ax.set_title('ECE (Probability Metrics)\nMacro-Averaged', fontweight='bold')
        ax.set_ylabel('ECE')
        ax.grid(True, alpha=0.3)
    
    # Fleiss Kappa Per-Client
    ax = axes[2, 0]
    cent_fleiss_pc = get_macro_avg(centralized_consistency, 'per_client', 'fleiss_kappa')
    local_fleiss_pc = get_macro_avg(local_consistency, 'per_client', 'fleiss_kappa')
    fed_fleiss_pc = get_macro_avg(federated_consistency, 'per_client', 'fleiss_kappa')
    data_to_plot = [v for v in [cent_fleiss_pc, local_fleiss_pc, fed_fleiss_pc] if v]
    labels = ['Centralized' if cent_fleiss_pc else '', 'Local' if local_fleiss_pc else '', 'Federated' if fed_fleiss_pc else '']
    labels = [l for l in labels if l]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
        ax.set_title('Fleiss Kappa (Per-Client Thresholds)\nMacro-Averaged', fontweight='bold')
        ax.set_ylabel('Fleiss Kappa')
        ax.set_ylim([-1, 1])
        ax.grid(True, alpha=0.3)
    
    # Fleiss Kappa Global
    ax = axes[2, 1]
    cent_fleiss_gl = get_macro_avg(centralized_consistency, 'global', 'fleiss_kappa')
    local_fleiss_gl = get_macro_avg(local_consistency, 'global', 'fleiss_kappa')
    fed_fleiss_gl = get_macro_avg(federated_consistency, 'global', 'fleiss_kappa')
    data_to_plot = [v for v in [cent_fleiss_gl, local_fleiss_gl, fed_fleiss_gl] if v]
    labels = ['Centralized' if cent_fleiss_gl else '', 'Local' if local_fleiss_gl else '', 'Federated' if fed_fleiss_gl else '']
    labels = [l for l in labels if l]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
        ax.set_title('Fleiss Kappa (Global Thresholds)\nMacro-Averaged', fontweight='bold')
        ax.set_ylabel('Fleiss Kappa')
        ax.set_ylim([-1, 1])
        ax.grid(True, alpha=0.3)
    
    # Empty subplot for alignment
    ax = axes[2, 2]
    ax.axis('off')
    
    plt.suptitle(f'Macro-Averaged Metrics Summary {DATASET}_{IID_TYPE}\n(Centralized vs Local vs Federated Learning)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'summary.pdf'}")
    
    print("\nAll visualizations saved successfully!")


if __name__ == "__main__":
    main()