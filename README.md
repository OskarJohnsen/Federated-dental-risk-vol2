# DIGIT Federated Recommenders

A federated learning system for predicting post-operative complications following wisdom tooth extraction. The project consists of two main components:

1. **Dataset Generation**: Synthetic medical data generation with configurable IID/non-IID partitioning
2. **Machine Learning Pipeline**: Training models using centralized, local, or federated learning paradigms

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/smoothyy3/DIGIT-Federated-Recommenders
cd DIGIT-Federated-Recommenders

# Install the package (recommended: use a virtual environment)
pip install -e .
```

### Setup

1. **Configure WandB** (required for training):
   ```bash
   wandb login
   export WANDB_PROJECT=your-project-name  # Optional, defaults to 'digit-federated-recommenders'
   ```
   See [SETUP.md](SETUP.md) for detailed setup instructions.

2. **Generate a dataset**:
   ```bash
   digit-fr-generate
   ```
   This creates a synthetic dataset and automatically generates a global test set.

3. **Train a model**:
   ```bash
   digit-fr-train centralized    # Centralized training
   digit-fr-train local          # Local (per-client) training
   digit-fr-train federated      # Federated learning
   ```

## Documentation

- **[SETUP.md](SETUP.md)** - Installation, configuration, and environment setup
- **[DATASET_GENERATION.md](docs/DATASET_GENERATION.md)** - Detailed guide to dataset generation system
- **[TRAINING.md](docs/TRAINING.md)** - Machine learning pipeline and training paradigms

## Project Structure

```
DIGIT-Federated-Recommenders/
├── configs/              # Configuration files (JSON)
├── data/                 # Generated datasets and results
│   ├── raw/             # Raw datasets (excluded from git)
│   ├── processed/       # Processed datasets (excluded from git)
│   └── results/         # Visualization outputs (tracked in git)
├── notebooks/            # Jupyter notebooks (EDA)
├── scripts/              # Utility scripts
│   ├── export_wandb_run.py    # Export WandB run to CSV
│   └── visualize_results.py   # Generate plots
├── src/digit_fr/
│   ├── data_generation/  # Dataset generation system
│   └── ml/              # Machine learning pipeline
└── pyproject.toml        # Package configuration
```

## Key Features

### Dataset Generation
- **Synthetic medical data** based on evidence-based rules
- **Configurable IID/non-IID partitioning** using NIID-Bench (Dirichlet distribution)
- **Automatic test set creation** and backup
- **Client-specific variations** (prevalence shifts, missingness, noise)

### Machine Learning
- **Three training paradigms**: Centralized, Local, Federated
- **Multi-task learning**: Trains on 4 binary risk outcomes, evaluates on risk categories
- **Risk categorization**: Uses percentile-based thresholds (33rd/67th) to convert probabilities to Low/Medium/High categories
- **Dual evaluation**:
  - **Probability metrics**: MSE/MAE comparing predicted vs. true risk probabilities
  - **Category metrics**: F1/Accuracy comparing predicted vs. true risk categories
- **Threshold strategies**: Global thresholds (from dataset) or per-client thresholds (from validation set)
- **WandB integration**: Experiment tracking and logging

## CLI Commands

### Dataset Generation
```bash
digit-fr-generate [OPTIONS]

Options:
  --seed INT              Random seed override
  --output-dir PATH       Output directory (default: from config)
  --formats TEXT          Output formats: csv,xlsx (default: csv,xlsx)
  --create-test-set       Create global test set (default: True)
  --test-samples INT      Number of test samples (default: 3000)
  --test-seed INT         Test set random seed (default: 999)
  --backup                Create backup of original dataset (default: True)
```

### Training
```bash
digit-fr-train {centralized|local|federated}
```

## Medical Domain

The system simulates federated learning across multiple dental clinics to predict four post-operative complications:

1. **Alveolar Osteitis (Dry Socket)** - Base incidence: 2%
2. **Secondary Infection** - Base incidence: 1.5%
3. **Nerve Dysesthesia** - Base incidence: 0.6% (mandibular only)
4. **Bleeding** - Base incidence: 0.08%

See [DATASET_GENERATION.md](docs/DATASET_GENERATION.md) for detailed medical domain information.

## Contributing

When working on this project:
- Use environment variables for WandB configuration (see [SETUP.md](SETUP.md))
- Follow the existing code structure and patterns
- Update documentation when adding new features
- Test changes with both IID and non-IID datasets

## License

#todo