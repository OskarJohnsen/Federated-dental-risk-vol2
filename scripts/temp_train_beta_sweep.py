"""
Temporary script to train models on beta sweep datasets where betaQ == betaL.

Trains all three paradigms (centralized, local, federated) for each beta value.

Usage:
    python scripts/temp_train_beta_sweep.py
"""

import sys
from pathlib import Path
from digit_fr.core.paths import root_path
from digit_fr.ml.config.experiment_config import ExperimentConfig
from digit_fr.ml.centralized.train import main as train_centralized
from digit_fr.ml.local.train import main as train_local
from digit_fr.ml.federated.train import main as train_federated

# Beta values where betaQ == betaL
BETA_VALUES = [0.1, 0.3, 0.5, 1.0, 5.0, 10.0]
DATASET = "A"
IID_TYPE = "non-iid"
MODEL_SEED = 42
DATA_SPLIT_SEED = 42

def train_beta_sweep():
    """Train all paradigms for each beta value."""
    
    for beta in BETA_VALUES:
        beta_str = str(beta).replace(".", "_")
        beta_name = f"betaL{beta}_betaQ{beta}"
        
        dataset_path = root_path('data', 'raw', 'sweep_beta', f'fed_recommenders_synthetic_dataset_{DATASET}_{IID_TYPE}_{beta_name}.csv')
        test_set_path = root_path('data', 'processed', 'A', 'sweep_beta', f'global_test_set_{IID_TYPE}_{beta_name}.csv')
        
        if not dataset_path.exists():
            print(f"Warning: Dataset not found: {dataset_path}", file=sys.stderr)
            continue
        
        if not test_set_path.exists():
            print(f"Warning: Test set not found: {test_set_path}", file=sys.stderr)
            continue
        
        experiment_id = f"{DATASET}_{IID_TYPE}_{beta_name}"
        
        print(f"\n{'='*80}")
        print(f"Training for beta={beta} ({beta_name})")
        print(f"{'='*80}")
        
        # Centralized
        print(f"\n--- CENTRALIZED TRAINING ---")
        try:
            config_centralized = ExperimentConfig(
                experiment_type="centralized",
                experiment_id=experiment_id,
                model_seed=MODEL_SEED,
                data_split_seed=DATA_SPLIT_SEED,
                dataset_path=str(dataset_path),
                test_set_path=str(test_set_path),
            )
            train_centralized(config_centralized)
        except Exception as e:
            print(f"Error in centralized training: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        
        # Local
        print(f"\n--- LOCAL TRAINING ---")
        try:
            config_local = ExperimentConfig(
                experiment_type="local",
                experiment_id=experiment_id,
                model_seed=MODEL_SEED,
                data_split_seed=DATA_SPLIT_SEED,
                dataset_path=str(dataset_path),
                test_set_path=str(test_set_path),
            )
            train_local(config_local)
        except Exception as e:
            print(f"Error in local training: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        
        # Federated
        print(f"\n--- FEDERATED TRAINING ---")
        try:
            config_federated = ExperimentConfig(
                experiment_type="federated",
                experiment_id=experiment_id,
                model_seed=MODEL_SEED,
                data_split_seed=DATA_SPLIT_SEED,
                dataset_path=str(dataset_path),
                test_set_path=str(test_set_path),
                federated_rounds=6,
                clients_per_round=None,
                local_epochs=5,
                aggregation_method="fedavg",
            )
            train_federated(config_federated)
        except Exception as e:
            print(f"Error in federated training: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        
        print(f"\nCompleted training for beta={beta}\n")


def main():
    print("Beta Sweep Training")
    print(f"Beta values: {BETA_VALUES}")
    print(f"Paradigms: centralized, local, federated")
    print(f"Total runs: {len(BETA_VALUES) * 3}")
    
    train_beta_sweep()
    
    print("\n" + "="*80)
    print("All training completed!")
    print("="*80)

if __name__ == "__main__":
    main()
