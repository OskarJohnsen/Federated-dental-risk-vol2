from __future__ import annotations
import typer
from .config.experiment_config import ExperimentConfig
from .constants import DATASET, IID_TYPE
from ..core.paths import root_path

app = typer.Typer(add_completion=False)

VALID_EXPERIMENT_TYPES = ["centralized", "local", "federated"]

@app.command()
def train(experiment_type: str = typer.Argument(), aggregation_method: str = typer.Option(
        "fedavg",
        "--aggregation-method",
        "-a",
        help="Aggregation method for federated training: fedavg | balanced",
        case_sensitive=False,
    ),
):
    if experiment_type == "centralized":
        from .centralized.train import main as train_centralized
        config = ExperimentConfig(
            experiment_type="centralized",
            experiment_id=f"{DATASET}_{IID_TYPE}",
            model_seed=42,
            data_split_seed=42,
            dataset_path=str(root_path('data', 'raw', f'synthetic_dataset_{DATASET}_{IID_TYPE}.csv')),
            test_set_path=str(root_path('data', 'processed', f'{DATASET}', f'global_test_set_{IID_TYPE}.csv')),
        )
        train_centralized(config)
    elif experiment_type == "local":
        from .local.train import main as train_local
        config = ExperimentConfig(
            experiment_type="local",
            experiment_id=f"{DATASET}_{IID_TYPE}",
            model_seed=42,
            data_split_seed=42,
            dataset_path=str(root_path('data', 'raw', f'synthetic_dataset_{DATASET}_{IID_TYPE}.csv')),
            test_set_path=str(root_path('data', 'processed', f'{DATASET}', f'global_test_set_{IID_TYPE}.csv')),
        )
        train_local(config)
    elif experiment_type == "federated":
        from .federated.train import main as train_federated

        valid_aggs = {"fedavg", "balanced"}
        agg = aggregation_method.lower()
        if agg not in valid_aggs:
            raise typer.BadParameter(f"aggregation_method must be one of {sorted(valid_aggs)}")

        config = ExperimentConfig(
            experiment_type="federated",
            experiment_id=f"{DATASET}_{IID_TYPE}",
            model_seed=42,
            data_split_seed=42,
            dataset_path=str(root_path('data', 'raw', f'synthetic_dataset_{DATASET}_{IID_TYPE}.csv')),
            test_set_path=str(root_path('data', 'processed', f'{DATASET}', f'global_test_set_{IID_TYPE}.csv')),
            federated_rounds=6,
            clients_per_round=None,
            local_epochs=5,
            aggregation_method=agg,
        )
        train_federated(config)

def run():
    app()

if __name__ == "__main__":
    run()