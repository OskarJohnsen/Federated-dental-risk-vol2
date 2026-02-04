# Setup Guide

This guide covers installation, configuration, and environment setup for the Federated Dental Risk Prediction project.

## Installation

### Prerequisites

- Python >= 3.10
- pip or conda

### Install Package

```bash
# Clone the repository
git clone https://github.com/smoothyy3/federated-dental-risk-prediction
cd federated-dental-risk-prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## WandB Configuration

WandB is used for experiment tracking. Each user should configure their own WandB account.

### Initial Setup

1. **Login to WandB**:
   ```bash
   wandb login
   ```
   Follow the prompts to authenticate with your WandB account.

2. **Configure Project Name** (optional):
   ```bash
   export WANDB_PROJECT=your-project-name
   ```
   Defaults to `federated-dental-risk-prediction` if not set.

3. **Configure Entity** (optional):
   ```bash
   export WANDB_ENTITY=your-username
   ```
   Defaults to your logged-in user if not set.

### Persistent Configuration

To make these settings persistent, add them to your shell configuration file:

**Bash/Zsh** (`~/.bashrc` or `~/.zshrc`):
```bash
export WANDB_PROJECT=your-project-name
export WANDB_ENTITY=your-username
```

**Windows (PowerShell)** (`$PROFILE`):
```powershell
$env:WANDB_PROJECT="your-project-name"
$env:WANDB_ENTITY="your-username"
```

### Verification

Test your WandB setup:
```bash
python -c "import wandb; print(f'Project: {wandb.api.default_project()}')"
```

## Dataset Configuration

The project uses constants defined in `src/fdrp/ml/constants.py`:
- `DATASET`: Dataset identifier (default: "A")
- `IID_TYPE`: IID or non-IID partitioning (default: "non-iid")

These are intentionally hardcoded for consistency across the codebase. To override, modify the constants file or use environment variables (if implemented).

## Project Structure

Key directories:
- `configs/` - JSON configuration files for generation and training
- `data/raw/` - Generated datasets (excluded from git)
- `data/processed/` - Processed datasets and test sets (excluded from git)
- `data/results/` - Visualization outputs (tracked in git)
- `checkpoints/` - Model checkpoints (excluded from git)
- `wandb/` - WandB logs (excluded from git)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_PROJECT` | WandB project name | `federated-dental-risk-prediction` |
| `WANDB_ENTITY` | WandB entity/username | Logged-in user |
| `FDRP_ROOT` | Project root directory | Auto-detected |

## Troubleshooting

### WandB Authentication Issues
- Ensure `wandb login` completed successfully
- Check `~/.netrc` contains your WandB API key
- Verify environment variables are set correctly

### Import Errors
- Ensure the package is installed: `pip install -e .`
- Check Python version: `python --version` (should be >= 3.10)
- Verify virtual environment is activated

### Path Resolution Issues
- The project auto-detects the root directory by looking for `pyproject.toml` and `configs/`
- If detection fails, set `FDRP_ROOT` environment variable to the project root

## Next Steps

- See [DATASET_GENERATION.md](docs/DATASET_GENERATION.md) for dataset generation guide
- See [TRAINING.md](docs/TRAINING.md) for training pipeline guide