from __future__ import annotations
import typer

app = typer.Typer(add_completion=False)

VALID_EXPERIMENT_TYPES = ["centralized", "local", "federated"]

@app.command()
def train(experiment_type: str = typer.Argument()):  
    if experiment_type == "centralized":
        from .centralized.train import main as train_centralized
        train_centralized()
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