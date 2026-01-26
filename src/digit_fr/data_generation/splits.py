"""
Dataset splitting utilities for creating test sets.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from ..core.paths import ensure_dir

def create_global_test_set(dataset_path: Path, output_path: Path, n_samples: int = 3000, seed: int = 999, backup_original: bool = True) -> Path:
    """
    Create a global test set by splitting a dataset.
    
    Args:
        dataset_path: Path to the full dataset CSV file
        output_path: Path where the test set CSV will be saved
        n_samples: Number of samples to include in test set
        seed: Random seed for reproducible splitting
        backup_original: Whether to create a backup of the original dataset
        
    Returns:
        Path to the created test set file
        
    Raises:
        FileNotFoundError: If dataset_path doesn't exist
        ValueError: If n_samples exceeds dataset size
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    print(f"Creating global test set")
    print(f"Dataset: {dataset_path}")
    print(f"Test samples: {n_samples:,}")
    print(f"Seed: {seed}")
    
    df_original = pd.read_csv(dataset_path)
    n_total = len(df_original)
    
    if n_samples > n_total:
        raise ValueError(f"Requested {n_samples} test samples, but dataset only has {n_total} samples")
    
    # Create backup if requested
    if backup_original:
        backup_path = dataset_path.parent / f"{dataset_path.stem}_backup{dataset_path.suffix}"
        print(f"Creating backup: {backup_path}")
        df_original.to_csv(backup_path, index=False)
    
    # Split dataset
    np.random.seed(seed)
    test_indices = np.random.choice(n_total, size=n_samples, replace=False)
    test_indices = np.sort(test_indices)
    
    df_test = df_original.iloc[test_indices].copy().reset_index(drop=True)
    train_val_mask = ~np.isin(np.arange(n_total), test_indices)
    df_remaining = df_original.iloc[train_val_mask].copy().reset_index(drop=True)
    
    print(f"Original dataset: {n_total:,} samples")
    print(f"Test set: {len(df_test):,} samples")
    print(f"Remaining (train&val): {len(df_remaining):,} samples")
    
    # Handle probability columns
    target_probabilities = [
        "Risk_AlveolarOsteitis_Prob",
        "Risk_SecondaryInfection_Prob",
        "Risk_NerveDysesthesia_Prob",
        "Risk_Bleeding_Prob"
    ]
    
    available_probs = [col for col in target_probabilities if col in df_test.columns]
    if available_probs:
        for col in available_probs:
            col_data = df_test[col]
            if col_data.max() > 1.0 and col_data.max() <= 100.0:
                print(f"  Scaling {col} from percentage to probability")
                df_test[col] = (col_data / 100.0).clip(0.0, 1.0)
        print(f"Included {len(available_probs)} probability columns")
    
    target_categories = [
        "Risk_Category_AlveolarOsteitis",
        "Risk_Category_SecondaryInfection",
        "Risk_Category_NerveDysesthesia",
        "Risk_Category_Bleeding"
    ]
    available_categories = [col for col in target_categories if col in df_test.columns]
    if available_categories:
        print(f"Included {len(available_categories)} risk category columns")
    
    ensure_dir(output_path.parent)
    df_test.to_csv(output_path, index=False)
    print(f"  Test set saved to: {output_path}")
    
    df_remaining.to_csv(dataset_path, index=False)
    print(f"Updated original dataset: removed {len(df_test):,} test samples")
    
    return output_path