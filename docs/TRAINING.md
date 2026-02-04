# Training Guide

This document provides a comprehensive guide to the machine learning training pipeline.

## Overview

The ML pipeline supports three training paradigms:
1. **Centralized**: All data in one place, standard supervised learning
2. **Local**: Per-client models trained independently
3. **Federated**: Federated learning using Flower framework

**Important**: Models are **trained on binary risk outcomes** (the Bernoulli-drawn outcomes), but **evaluated on risk categories** (Low/Medium/High severity levels). This is because the Bernoulli draw introduces significant noise, especially for rare events, making it difficult for models to predict exact small percentages. Risk categories provide a more stable and clinically meaningful evaluation target.

## Quick Start

```bash
# Centralized training
fdrp-train centralized

# Local (per-client) training
fdrp-train local

# Federated learning
fdrp-train federated
```

## Prerequisites

1. **Dataset**: Generate a dataset first (see [DATASET_GENERATION.md](DATASET_GENERATION.md))
2. **WandB**: Configure WandB (see [SETUP.md](../SETUP.md))
3. **Dependencies**: All dependencies installed via `pip install -e .`

## Model Architecture

### MLP (Multi-Layer Perceptron)

**Location**: `src/fdrp/ml/models/architectures/mlp.py`

**Architecture**:
- **Input Layer**: Size = number of features (after preprocessing, typically ~30-35)
- **Hidden Layers**: Configurable (default: [128, 64] neurons)
  - Each hidden layer: Linear → BatchNorm1d → ReLU → Dropout
- **Output Layer**: 4 independent binary classification heads (one per risk type)
  - Each head: Linear layer → sigmoid activation → binary prediction

**Key Design Decisions**:
- **Multi-task Learning**: Single shared feature extractor with 4 task-specific heads
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout**: Prevents overfitting (default: 0.2)
- **Independent Heads**: Each risk type predicted independently (no shared output layer)

## Training Pipeline

### Step 1: Data Loading (`src/fdrp/ml/data/loaders.py`)

1. **Load Raw Data**: Read CSV file
2. **Feature/Target Separation**:
   - **Features (X)**: All columns except targets and leakage variables
   - **Leakage Prevention**: Excludes:
     - `Patient`, `Client` (identifiers)
     - `Removal_Prob`, `Score_*`, `Prob_*` (decision process information)
     - `Risk_*_Prob` (true risk probabilities - leakage)
     - `Risk_Category_*` (all risk category columns including `Risk_Category_Composite` - leakage, as they're derived from risk probabilities)
   - **Training Targets (y)**: 4 binary risk columns (`Risk_AlveolarOsteitis`, etc.) - used for training loss
   - **Evaluation Targets**: 
     - Risk probabilities (`Risk_*_Prob`) - for MSE/MAE evaluation
     - Risk categories (`Risk_Category_*`) - for F1/accuracy evaluation
3. **Missing Data Indicators**: Creates binary indicators for missing values in `Tooth_Mobility` and `Bone_Density`
4. **Train/Val/Test Split**: 
   - Default: 60% train, 20% val, 20% test
   - Uses `sklearn.model_selection.train_test_split` with random seed
5. **Categorical Encoding**: One-hot encoding for `Surgical_Extraction_Type` and `Tooth_Angulation`
6. **Imputation**: Median imputation for missing values (using `SimpleImputer`)

### Step 2: Model Initialization

**Location**: `src/fdrp/ml/centralized/train.py` (lines 55-60)

- Creates MLP with configurable architecture
- Input size determined from preprocessed data
- 4 classification heads (one per risk type)

### Step 3: Loss Function and Optimization

**Loss Function**: `BCEWithLogitsLoss` (Binary Cross-Entropy with Logits)
- Applied independently to each of the 4 risk predictions
- **Class Weights**: Optional (`use_class_weights=True`)
  - Computes positive class weights: `weight = neg_count / pos_count`
  - Clamped between 1.0 and 100.0 to prevent extreme weights
  - Addresses class imbalance

**Optimizer**: Adam or AdamW (configurable)
- Learning rate: 1e-4 (default)
- Weight decay: 1e-5 (default)

**Scheduler**: Optional StepLR
- Reduces learning rate by factor `gamma` every `step_size` epochs
- Default: step_size=5, gamma=0.5

### Step 4: Training Loop (`src/fdrp/ml/models/base/trainer.py`)

**BaseTrainer.fit()**:
1. **Forward Pass**: Model predicts logits for 4 risks
2. **Loss Calculation**: BCEWithLogitsLoss on predictions vs. true labels
3. **Backward Pass**: Gradient computation and optimizer step
4. **Validation**: After each epoch, evaluate on validation set
5. **History Tracking**: Stores train/val loss per epoch
6. **Checkpointing**: Saves model state after training

### Step 5: Threshold Optimization

**Problem**: Binary classification with imbalanced classes requires optimal thresholds (not always 0.5)

**Methods** (`src/fdrp/ml/metrics/threshold.py`):

1. **Youden's J Index** (default):
   - Maximizes: `J = Sensitivity + Specificity - 1`
   - Finds threshold that maximizes the sum of true positive rate and true negative rate
   - Good for balanced sensitivity/specificity

2. **F1 Score Optimization**:
   - Maximizes F1 score: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
   - Good when precision and recall are both important

**Process**:
- Evaluate model on validation set to get probability predictions
- For each risk type, test thresholds from 0.0 to 1.0
- Select threshold that optimizes the chosen metric
- Apply optimized thresholds to test set predictions

### Step 6: Evaluation (`src/fdrp/ml/models/base/trainer.py`)

**Important**: Models are **trained on binary risk outcomes**, but **evaluated on risk categories**. This is because the Bernoulli draw introduces too much noise for models to accurately predict the exact small percentages.

**BaseTrainer.evaluate()**:
1. **Inference**: Model in eval mode, no gradients
2. **Probability Conversion**: Apply sigmoid to logits → probabilities
3. **Evaluation Metrics** (`src/fdrp/ml/metrics/calc_metrics.py`):

   **A. Probability Metrics** (comparing predicted probabilities to true risk probabilities):
   - **MSE** (Mean Squared Error): Per-risk and macro-averaged
   - **MAE** (Mean Absolute Error): Per-risk and macro-averaged
   - **Brier Score**: Calibration metric for probabilities
   - **ECE** (Expected Calibration Error): Calibration quality

   **B. Category Metrics** (comparing predicted categories to true categories):
   - **F1 Score**: Per-risk and macro-averaged
   - **Accuracy**: Per-risk and macro-averaged
   - Categories are computed by applying thresholds to predicted probabilities

4. **Threshold Strategies**:
   - **Global Thresholds**: Use percentile boundaries from dataset generation (33rd/67th percentiles)
   - **Per-Client Thresholds**: Compute percentile boundaries from each client's validation set
   - Both strategies can be evaluated simultaneously (`category_strategy="both"`)

### Step 7: Risk Categorization and Evaluation

**Risk Categorization** (`src/fdrp/ml/metrics/threshold.py`):

After model inference, predicted probabilities are converted to categories using thresholds:

1. **Apply Thresholds**: Use percentile boundaries (33rd/67th) to categorize probabilities:
   - Low (0): probability < 33rd percentile
   - Medium (1): 33rd percentile ≤ probability < 67th percentile
   - High (2): probability ≥ 67th percentile

2. **Threshold Sources**:
   - **Global**: Loaded from `configs/global_thresholds/{DATASET}/global_thresholds_{IID_TYPE}.json` (computed during dataset generation)
   - **Per-Client**: Computed from validation set probabilities using `percentile_thresholds()` function

3. **Category Evaluation** (`src/fdrp/ml/metrics/calc_metrics.py`):
   - Compare predicted categories to true categories (from dataset)
   - Compute F1 score and accuracy per risk type
   - Compute macro-averaged metrics across all risks

**Why This Approach?**
- Binary outcomes (Bernoulli-drawn) are too noisy, especially for rare events (e.g., Bleeding: 0.08% base)
- Models can learn dataset distributions but struggle with exact small percentages
- Risk categories provide stable, clinically meaningful evaluation targets
- Categories represent actionable risk severity levels

### Step 8: Logging and Reporting

**WandB Integration** (`src/fdrp/ml/metrics/report.py`):
- Logs experiment configuration (hyperparameters, data version, code version)
- Logs training history (loss per epoch)
- Logs dataset statistics (class distributions, imbalance ratios)
- Logs probability metrics (MSE, MAE, Brier Score, ECE)
- Logs category metrics (F1, Accuracy) for both global and per-client thresholds
- Logs optimized thresholds (for binary classification, though not used in final evaluation)

## Training Paradigms

### Centralized Training

**Command**: `fdrp-train centralized`

**Description**: Standard supervised learning where all training data is available in one place.

**Process**:
1. Load full dataset
2. Split into train/val/test
3. Train single model on all training data
4. Evaluate on global test set

**Use Case**: Baseline comparison, upper bound performance

### Local Training

**Command**: `fdrp-train local`

**Description**: Each client trains their own model independently on their local data.

**Process**:
1. Load full dataset
2. Split data by client
3. For each client:
   - Create train/val/test splits
   - Train independent model
   - Evaluate on global test set
4. Aggregate metrics across clients

**Use Case**: Lower bound comparison, individual client performance

### Federated Learning

**Command**: `fdrp-train federated`

**Description**: Federated learning using Flower framework. Multiple clients collaborate to train a shared model without sharing raw data.

**Process**:
1. Initialize global model
2. For each federated round:
   - Select clients (all or subset)
   - Each client trains locally for `local_epochs`
   - Aggregate model updates (FedAvg)
   - Update global model
3. Evaluate final global model on global test set

**Configuration** (in `src/fdrp/ml/cli.py`):
- `federated_rounds`: Number of federated rounds (default: 6)
- `clients_per_round`: Number of clients per round (default: None = all clients)
- `local_epochs`: Epochs per client per round (default: 5)
- `aggregation_method`: Aggregation strategy (default: "fedavg")

**Use Case**: Realistic federated learning scenario

## Experiment Configuration

**Location**: `src/fdrp/ml/config/experiment_config.py`

**ExperimentConfig** dataclass contains:
- **Experiment Type**: `centralized`, `local`, or `federated`
- **Seeds**: Separate seeds for data splitting and model initialization
- **Data Splits**: Train/val/test sizes, split strategy
- **Model**: Architecture type, hidden sizes, dropout
- **Training**: Batch size, learning rate, epochs, optimizer, scheduler
- **Loss**: Loss function, class weights
- **Evaluation**: Threshold optimization method
- **Versioning**: Data version (MD5 hash), code version (git commit)

**Current Configuration** (hardcoded in CLI):
- Model seed: 42
- Data split seed: 42
- Dataset path: `data/raw/synthetic_dataset_{DATASET}_{IID_TYPE}.csv`
- Test set path: `data/processed/{DATASET}/global_test_set_{IID_TYPE}.csv`

## Code Structure

```
src/fdrp/ml/
├── centralized/          # Centralized training
│   └── train.py         # Main training script
├── federated/            # Federated learning
│   ├── train.py         # Federated training script
│   └── aggregation.py   # Aggregation strategies
├── local/                # Local training (per-client models)
│   └── train.py         # Local training script
├── config/              # Experiment configuration
│   └── experiment_config.py  # ExperimentConfig dataclass
├── constants.py          # Risk names and constants
├── data/                 # Data loading and preprocessing
│   ├── loaders.py       # Data loading with train/val/test splits
│   ├── datasets.py      # PyTorch Dataset classes
│   └── preprocessing.py # Preprocessing pipeline
├── metrics/              # Evaluation metrics
│   ├── calc_metrics.py  # Metric calculations (precision, recall, F1, etc.)
│   ├── threshold.py     # Threshold optimization (F1, Youden's J)
│   └── report.py        # WandB logging and reporting
├── models/               # Model architectures and training
│   ├── architectures/
│   │   └── mlp.py       # Multi-Layer Perceptron implementation
│   └── base/
│       ├── model.py     # BaseModel abstract class
│       └── trainer.py   # BaseTrainer class (training loop, evaluation)
└── util/
    ├── seed.py          # Seed management for reproducibility
    └── wandb_config.py  # WandB configuration utilities
```

## Evaluation Approach: Why Categories Instead of Binary Outcomes?

**Training**: Models are trained on binary risk outcomes (Bernoulli-drawn from risk probabilities).

**Evaluation**: Models are evaluated on **risk categories**, not binary outcomes. This is a critical design decision:

### The Problem with Binary Outcomes

1. **High Noise**: Bernoulli draw introduces significant stochasticity, especially for rare events
   - Example: Bleeding has 0.08% base incidence → very few positive samples
   - Even with perfect probability prediction, binary outcomes are highly variable

2. **Learning Limitation**: Models can learn the dataset distribution but struggle with exact small percentages
   - A model predicting 0.5% risk correctly may still get many binary predictions wrong due to randomness
   - Binary metrics (precision, recall, F1) become unreliable for imbalanced, rare events

### The Solution: Risk Categories

1. **Stable Targets**: Categories (Low/Medium/High) are computed from risk probabilities using percentiles
   - More stable than binary outcomes
   - Less affected by Bernoulli noise

2. **Clinically Meaningful**: Categories represent actionable risk severity levels
   - Low risk: < 33rd percentile
   - Medium risk: 33rd-67th percentile
   - High risk: ≥ 67th percentile

3. **Dual Evaluation**:
   - **Probability Metrics (MSE/MAE)**: Assess how well models predict true risk probabilities
   - **Category Metrics (F1/Accuracy)**: Assess how well models categorize patients into risk levels

### Threshold Strategies

- **Global Thresholds**: Use percentile boundaries from dataset generation (consistent across all clients)
- **Per-Client Thresholds**: Compute percentile boundaries from each client's validation set (accounts for client-specific distributions)

Both strategies can be evaluated simultaneously to understand model performance under different thresholding approaches.

## Key Design Patterns

1. **Base Classes**: `BaseModel` and `BaseTrainer` provide common interface for different architectures and training modes
2. **Modular Metrics**: Separate modules for calculation, threshold optimization, and reporting
3. **Reproducibility**: Comprehensive seed management and version tracking
4. **Leakage Prevention**: Careful exclusion of decision-related features from training data
5. **Multi-task Learning**: Single model predicts all 4 risks simultaneously
6. **Category-Based Evaluation**: Evaluation on risk categories rather than noisy binary outcomes

## Visualization and Analysis

After training, use the visualization scripts to analyze results:

1. **Export WandB Run**:
   ```bash
   python scripts/export_wandb_run.py <run_id>
   ```
   Exports metrics to CSV: `data/results/A/{iid_type}/wandb_export_*.csv`

2. **Visualize Results**:
   ```bash
   python scripts/visualize_results.py <csv_path>
   ```
   Generates publication-quality plots: `data/results/A/{iid_type}/*.pdf`

## Troubleshooting

### Dataset Not Found
- Ensure dataset is generated: `fdrp-generate`
- Check dataset path matches `DATASET` and `IID_TYPE` constants
- Verify file exists: `data/raw/synthetic_dataset_{DATASET}_{IID_TYPE}.csv`

### WandB Errors
- Ensure WandB is configured: `wandb login`
- Check environment variables: `echo $WANDB_PROJECT`
- Verify project/entity are accessible

### CUDA/GPU Issues
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Models will fall back to CPU if CUDA unavailable

### Memory Issues
- Reduce batch size in experiment config
- Use smaller dataset or fewer clients (for federated)

## Next Steps

- See [DATASET_GENERATION.md](DATASET_GENERATION.md) for dataset generation
- See [SETUP.md](../SETUP.md) for environment setup