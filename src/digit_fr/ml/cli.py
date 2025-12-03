from __future__ import annotations
import typer
from .config.experiment_config import ExperimentConfig
from ...core.paths import root_path

app = typer.Typer(add_completion=False)

VALID_EXPERIMENT_TYPES = ["centralized", "local", "federated"]

@app.command()
def train(experiment_type: str = typer.Argument()):  
    if experiment_type == "centralized":
        from .centralized.train import main as train_centralized
        config = ExperimentConfig(
            experiment_type="centralized",
            experiment_id="baseline_v1",
            model_seed=42,
            data_split_seed=42,
            dataset_path=str(root_path('data', 'raw', 'fed_recommenders_synthetic_dataset_50k.csv')),
            test_set_path=str(root_path('data', 'processed', 'global_test_set.csv')),
        )
        train_centralized(config)
    elif experiment_type == "local":
        typer.echo("Local training not yet implemented")
        raise typer.Exit(code=1)
    elif experiment_type == "federated":
        typer.echo("Federated training not yet implemented")
        raise typer.Exit(code=1)

def run():
    app()

if __name__ == "__main__":
    run()