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

The project uses constants defined in `src/fdrp/ml/constants.py` that control file paths, experiment IDs, and data organization:

### Constants Overview

- **`RISK_NAMES`**: List of the four risk types being predicted
  - `["AlveolarOsteitis", "SecondaryInfection", "NerveDysesthesia", "Bleeding"]`
  - Used throughout the codebase for iterating over risks, computing metrics, and generating column names
  - **Do not modify** unless you're changing the medical domain

- **`DATASET`**: Dataset identifier (default: `"A"`)
  - Used in file paths to organize datasets and results
  - Affects paths like:
    - `data/raw/synthetic_dataset_{DATASET}_{IID_TYPE}.csv`
    - `data/processed/{DATASET}/global_test_set_{IID_TYPE}.csv`
    - `configs/global_thresholds/{DATASET}/global_thresholds_{IID_TYPE}.json`
    - `data/results/{DATASET}/{IID_TYPE}/`
  - Change this if you want to organize multiple dataset variants (different client size etc.)

- **`IID_TYPE`**: IID or non-IID partitioning (default: `"non-iid"`)
  - Controls which dataset variant is used: `"iid"` or `"non-iid"`
  - Used in the same file paths as `DATASET` above
  - Also used in experiment IDs: `{DATASET}_{IID_TYPE}`
  - Change this to switch between IID and non-IID experiments

### Where These Constants Are Used

1. **Dataset Generation** (`fdrp-generate`):
   - Output file names: `synthetic_dataset_{DATASET}_{IID_TYPE}.csv`
   - Test set paths: `data/processed/{DATASET}/global_test_set_{IID_TYPE}.csv`
   - Threshold file paths: `configs/global_thresholds/{DATASET}/global_thresholds_{IID_TYPE}.json`

2. **Training** (`fdrp-train`):
   - Dataset loading paths
   - Experiment IDs for WandB logging
   - Results directory organization

3. **Visualization** (`scripts/visualize_results.py`):
   - Results directory paths: `data/results/{DATASET}/{IID_TYPE}/`
   - Risk name iteration for plotting

### Changing Constants

These constants are intentionally hardcoded for consistency across the codebase. To change them:

1. **Edit `src/fdrp/ml/constants.py`** directly:
   ```python
   DATASET = 'B'  # Change from 'A' to 'B'
   IID_TYPE = 'iid'  # Change from 'non-iid' to 'iid'
   ```

2. **Ensure matching files exist**:
   - If changing `DATASET` or `IID_TYPE`, make sure the corresponding dataset files exist
   - Generate new datasets with `fdrp-generate` if needed

3. **Note**: Changing these constants affects **all** commands (generate, train, visualize), so ensure consistency across your workflow.

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