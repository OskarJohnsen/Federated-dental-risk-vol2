from ...core.paths import root_path, ensure_dir
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from ..util.seed import data_seeds
from typing import Optional
from ..config.experiment_config import ExperimentConfig
from .preprocessing import PreprocessingPipeline

def load_raw_data(dataset_path: Optional[str] = None):
    if dataset_path is None:
        dataset_path = root_path('data', 'raw', 'fed_recommenders_synthetic_dataset_50k.csv')
    df = pd.read_csv(dataset_path)
    
    target_classification = ["Risk_AlveolarOsteitis", "Risk_SecondaryInfection", "Risk_NerveDysesthesia", "Risk_Bleeding"]
    target_categories = ["Risk_Category_AlveolarOsteitis", "Risk_Category_SecondaryInfection", "Risk_Category_NerveDysesthesia", "Risk_Category_Bleeding"]
    leakage_cols = ["Patient", "Client", "Removal_Prob", "Score_1", "Score_2", "Score_3", "Prob_1", "Prob_2", "Prob_3", "Risk_AlveolarOsteitis_Prob", "Risk_SecondaryInfection_Prob", "Risk_NerveDysesthesia_Prob", "Risk_Bleeding_Prob"]
    
    target_probabilities = ["Risk_AlveolarOsteitis_Prob", "Risk_SecondaryInfection_Prob", "Risk_NerveDysesthesia_Prob", "Risk_Bleeding_Prob"]

    all_target_cols = target_classification + target_categories
    X = df.drop(columns=leakage_cols + all_target_cols)
    y_classification = df[target_classification]
    
    result = {'X': X, 'y_classification': y_classification, 'Client': df['Client']}
    
    available_probs = [col for col in target_probabilities if col in df.columns]
    if available_probs:
        y_probabilities = df[target_probabilities].copy()
        for col in y_probabilities.columns:
            y_probabilities[col] = y_probabilities[col].clip(0.0, 1.0)
        result['y_probabilities'] = y_probabilities
    
    return result

def load_global_test_set(test_set_path: Optional[str] = None, preprocessing_pipeline: Optional[PreprocessingPipeline] = None) -> dict:
    if test_set_path is None:
        FileNotFoundError("Need testset path")
    else:
        test_set_path = Path(test_set_path)
    
    if not test_set_path.exists():
        raise FileNotFoundError(
            f"Global test set not found at: {test_set_path}"
        )
    
    print(f"Loading global test set from: {test_set_path}")
    df = pd.read_csv(test_set_path)
    
    target_classification = ["Risk_AlveolarOsteitis", "Risk_SecondaryInfection", "Risk_NerveDysesthesia", "Risk_Bleeding"]
    target_categories = ["Risk_Category_AlveolarOsteitis", "Risk_Category_SecondaryInfection", "Risk_Category_NerveDysesthesia", "Risk_Category_Bleeding"]
    target_probabilities = ["Risk_AlveolarOsteitis_Prob", "Risk_SecondaryInfection_Prob", "Risk_NerveDysesthesia_Prob", "Risk_Bleeding_Prob"]

    optional_cols = ['Client'] + target_probabilities
    
    all_target_cols = target_classification + target_categories
    feature_cols = [col for col in df.columns if col not in all_target_cols + optional_cols]
    X_test = df[feature_cols].copy()
    y_test = df[target_classification].copy()
    
    if preprocessing_pipeline is not None:
        print("Applying preprocessing pipeline (to test set)")
        X_test = preprocessing_pipeline.transform(X_test)
        print(f"Preprocessed test set has: {len(X_test.columns)} features")
    
    result = {
        'X': X_test,
        'y_classification': y_test
    }
    
    if 'Client' in df.columns:
        result['Client'] = df['Client'].copy()
    
    available_categories = [col for col in target_categories if col in df.columns]
    if available_categories:
        result['y_categories'] = df[target_categories].copy()
        print(f"risk category columns for evaluation: {available_categories}")
    
    available_probs = [col for col in target_probabilities if col in df.columns]
    if available_probs:
        result['y_probabilities'] = df[target_probabilities].copy()
        for col in result['y_probabilities'].columns:
            result['y_probabilities'][col] = result['y_probabilities'][col].clip(0.0, 1.0)
        print(f"probability columns for evaluation: {available_probs}")
    else:
        print(f"PROB COLS NOT FOUND IN GLOBAL TEST")
    
    print(f"Test samples: {len(X_test):,}, features: {len(X_test.columns)}")
    return result


def load_data_with_split(test_size=0.2, val_size=0.2, data_split_seed=42, config: Optional[ExperimentConfig] = None):
    if config is not None:
        data_split_seed = config.data_split_seed
        test_size = config.test_size
        val_size = config.val_size
        dataset_path = config.dataset_path
    else:
        dataset_path = None

    data_seeds(data_split_seed)

    data = load_raw_data(dataset_path=dataset_path)

    client_ids = data['Client'].copy()

    val_size_adjusted = val_size / (1 - test_size)
    indices = data['X'].index
    train_indices, val_indices = train_test_split(indices, test_size=val_size_adjusted, random_state=data_split_seed)
    
    X_train = data['X'].loc[train_indices].copy()
    X_val = data['X'].loc[val_indices].copy()
    y_train = data['y_classification'].loc[train_indices].copy()
    y_val = data['y_classification'].loc[val_indices].copy()
    
    client_train = client_ids.loc[train_indices].copy()
    client_val = client_ids.loc[val_indices].copy()
    
    pipeline = PreprocessingPipeline()
    pipeline.fit(X_train)
    X_train_processed = pipeline.transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    
    result = {
        'train': {'X': X_train_processed, 'y_classification': y_train, 'Client': client_train},
        'val': {'X': X_val_processed, 'y_classification': y_val, 'Client': client_val},
        '_preprocessing_pipeline': pipeline
    }
    
    return result

def load_data_per_client(full_data: dict, client_id: int, config: Optional[ExperimentConfig] = None):
    if config is not None:
        data_split_seed = config.data_split_seed
        test_size = config.test_size
        val_size = config.val_size

    data_seeds(data_split_seed)

    client_mask = full_data['Client'] == client_id
    X_client = full_data['X'][client_mask].copy()
    y_client = full_data['y_classification'][client_mask].copy()
    
    if len(X_client) == 0:
        raise ValueError(f"No data found for client {client_id}")
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_client, y_client, test_size=val_size_adjusted, random_state=data_split_seed)
    
    pipeline = PreprocessingPipeline()
    pipeline.fit(X_train)
    X_train_processed = pipeline.transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    
    result = {
        'train': {'X': X_train_processed, 'y_classification': y_train},
        'val': {'X': X_val_processed, 'y_classification': y_val},
        '_preprocessing_pipeline': pipeline
    }
    
    return result