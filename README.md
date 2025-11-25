# DIGIT Federated Recommenders - Technical Documentation

## Table of Contents
1. [Medical Domain Overview](#medical-domain-overview)
2. [Dataset Generation System](#dataset-generation-system)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
4. [Project Structure](#project-structure)

---

## Medical Domain Overview

### Context: Wisdom Tooth Extraction

This project simulates a **federated learning** scenario for predicting post-operative complications following wisdom tooth (third molar) extraction. The system generates synthetic patient data across multiple dental clinics (clients) and trains machine learning models to predict four types of surgical risks.

### Risk Types (Target)

The system predicts four binary risk outcomes:

#### 1. **Alveolar Osteitis (Dry Socket)**
- **Medical Definition**: A painful condition where the blood clot at the extraction site dissolves prematurely, exposing the underlying bone and nerves.
- **Base Incidence**: 2% (0.02)
- **Clinical Incidence Range**: 3.4-6.3% in literature
- **Key Risk Factors**:
  - **Impaction Depth**: Deeper impactions (levels 2-3) increase risk (multipliers: 1.3, 1.4)
  - **Pericoronitis**: Active inflammation increases risk (multiplier: 1.4)
  - **Age**: Peak risk at 25-34 years (multiplier: 1.8), lower in younger patients (12-24: 0.8)
  - **Surgical Type**: Different extraction approaches modify risk (Type 1: 0.70, Type 2: 1.95, Type 3: 1.45)
- **Interaction Effects**: Deep impaction (level 3) combined with pericoronitis multiplies risk by 3.0

#### 2. **Secondary Infection**
- **Medical Definition**: Post-operative bacterial infection at the extraction site, potentially spreading to surrounding tissues.
- **Base Incidence**: 1.5% (0.015)
- **Clinical Incidence Range**: 1-5%
- **Key Risk Factors**:
  - **Impaction Depth**: Complete bony impaction (level 3) significantly increases risk (multiplier: 2.5)
  - **Cyst Presence**: Associated cysts increase risk (multiplier: 1.5)
  - **Periodontal Status**: Poor periodontal health (levels 2-3) increases risk (multipliers: 1.5, 2.5)
  - **Mandibular vs Maxillary**: Mandibular extractions have higher risk (multiplier: 1.3)
- **Interaction Effects**: Complete bony impaction (level 3) with diabetes multiplies risk by 3.0

#### 3. **Nerve Dysesthesia (Inferior Alveolar Nerve Injury)**
- **Medical Definition**: Temporary or permanent damage to the inferior alveolar nerve (IAN), causing numbness, tingling, or altered sensation in the lower lip, chin, and teeth.
- **Base Incidence**: 0.6% (0.006)
- **Clinical Incidence Range**: 0.4-0.6%
- **Key Risk Factors**:
  - **Impaction Depth**: Deep impactions (level 3) dramatically increase risk (multiplier: 3.0)
  - **Nerve Proximity**: Close proximity to IAN on radiographs (multiplier: 2.5)
  - **Tooth Angulation**: Mesioangular tilt (level 2) increases risk (multiplier: 2.0)
  - **Age**: Older patients (35+) have significantly higher risk (multiplier: 2.3)
  - **Mandibular Only**: This risk only applies to mandibular teeth (maxillary: risk = 0.0)
- **Special Logic**: 
  - If nerve proximity is absent (0), risk is reduced by 70% (multiplier: 0.30)
  - Maxillary teeth have zero risk for this outcome
- **Interaction Effects**: Deep impaction (level 3) with close nerve proximity multiplies risk by 3.0

#### 4. **Bleeding**
- **Medical Definition**: Excessive post-operative or intra-operative bleeding requiring intervention.
- **Base Incidence**: 0.08% (0.0008)
- **Clinical Incidence Range**: 0.2-0.6% post-op, up to 4.8% intra-op in older patients
- **Key Risk Factors**:
  - **Impaction Depth**: Moderate to deep impactions (levels 2-3) increase risk (multipliers: 1.9, 3.5)
  - **Clotting Disorders**: Presence of clotting disorders dramatically increases risk (multiplier: 3.8)
  - **Age**: Patients 36+ have significantly higher risk (multiplier: 3.5)
  - **Mandibular vs Maxillary**: Mandibular extractions have higher risk (multiplier: 1.7)
- **Interaction Effects**: Deep impaction (level 3) in patients 36+ multiplies risk by 5.0

---

## Dataset Generation System

### Overview

The dataset generator (`src/digit_fr/data_generation/`) creates synthetic patient records across multiple dental clinics (clients) in a federated learning setup. The generation process simulates realistic clinical decision-making and risk outcomes based on evidence-based medical rules.

### Generation Pipeline

#### Step 1: Patient Feature Generation

For each client (clinic), the system generates patient features using probabilistic models:

1. **Demographics**: Age (normal distribution, μ=28, σ=7, clipped 16-60), Sex (50/50), Mandibular/Maxillary (48/52)
2. **Symptoms**: Binary features with fixed probabilities (Pain: 50%, Swelling: 30%, Trismus: 20%, Pericoronitis: 40%)
3. **Anatomical Features**: Categorical distributions based on clinical prevalence
4. **Systemic Factors**: 
   - Conditional probabilities (e.g., osteoporosis depends on age and sex)
   - Bisphosphonates depend on osteoporosis status and age

#### Step 2: Client-Specific Variations

Each client can have:
- **Prevalence Shifts**: Different age distributions, nerve proximity rates, impaction depth distributions
- **Score Scaling**: Multipliers for surgical decision scores (simulating different clinical preferences)
- **Missingness**: Missing data rates for specific features (simulating data quality heterogeneity)
- **Feature Noise**: Measurement noise applied to features (simulating inter-clinician variability)

#### Step 3: Surgical Decision Generation

The system uses a **rule-based scoring system** with softmax selection:

1. **Base Priors**: Initial scores for each extraction type (Type 1: 0.30, Type 2: 0.50, Type 3: 0.20)
2. **Rule Application**: Multiple rule categories modify scores:
   - **Symptom Rules**: Pain, swelling, trismus, pericoronitis increase Type 2 scores
   - **Anatomy Rules**: Impaction depth, angulation, mandibular/maxillary location
   - **Pathology Rules**: Cysts, caries
   - **Systemic Rules**: Diabetes, osteoporosis, clotting disorders favor Type 3 (nerve-sparing)
   - **IAN Proximity Rules**: Close nerve proximity strongly favors Type 3
   - **Interactions**: Complex combinations (e.g., swelling + trismus + nerve proximity)
3. **Client Scaling**: Client-specific score multipliers applied
4. **Noise Injection**: Gaussian noise (σ=0.09) added to scores
5. **Softmax Selection**: Temperature-scaled softmax (temperature=1.0) converts scores to probabilities
6. **Decision Sampling**: Random selection based on computed probabilities

**Code Location**: `src/digit_fr/data_generation/rules/decision/extraction.py`

#### Step 4: Risk Percentage Calculation (Target)

For each of the four risk types, the system computes a **risk probability** using a multiplicative model:

**Formula**:
```
risk = base_incidence × ∏(risk_modifiers) × ∏(interactions) × surgery_modifier
```

**Process** (`src/digit_fr/data_generation/rules/decision/risk.py`):

1. **Initialize**: Start with `base_incidence` from `configs/risk_stats.json`
2. **Apply Risk Modifiers**: For each feature in `risk_modifiers`:
   - Look up the patient's feature value
   - Multiply risk by the corresponding modifier value
   - Example: If `Impaction_Depth=3` and modifier is `{"3": 2.5}`, multiply by 2.5
3. **Apply Interactions**: For each interaction rule:
   - Check if all conditions match the patient
   - If matched, apply interaction multiplier (with adjustment to avoid double-counting)
   - Example: Deep impaction + pericoronitis → multiply by 3.0
4. **Apply Surgery Modifier**: Based on the selected `Surgical_Extraction_Type`:
   - Look up modifier for that surgery type
   - Multiply risk by surgery modifier
5. **Special Cases**: 
   - **Nerve Dysesthesia**: Set to 0.0 for maxillary teeth
   - **Nerve Dysesthesia**: If nerve proximity is absent, multiply by 0.30
6. **Clamp**: Ensure risk is between 0.0 and 1.0

**Example Calculation** (Alveolar Osteitis with worst case symptoms):
```
base_incidence = 0.02
Impaction_Depth = 3 → multiplier = 1.4
Pericoronitis = 1 → multiplier = 1.4
Age = 30 → multiplier = 1.8 (age range 25-34)
Interaction (Deep+Pericoronitis) → multiplier = 3.0
Surgical_Extraction_Type = 2 → multiplier = 1.95

risk = 0.02 × 1.4 × 1.4 × 1.8 × 3.0 × 1.95 = 0.412 (41.2%)
```

#### Step 5: Bernoulli Draw (Binary Outcome Generation)

After computing the risk probability, the system performs a **Bernoulli trial** to determine the binary outcome:

**Process** (`src/digit_fr/data_generation/generation/synth.py`, lines 155-162):
```python
alveolar_risk = compute_risk_from_evidence(row, "AlveolarOsteitis", configs["risks"], d)
alveolar_risk_binary = int(np.random.rand() < alveolar_risk)
```

**Mathematical Description**:
- Generate a uniform random number `u ~ Uniform(0, 1)`
- If `u < risk_probability`, outcome = 1 (risk occurred)
- Otherwise, outcome = 0 (risk did not occur)

**Why Bernoulli?**
- The risk probability represents the **true probability** of the event occurring
- Each patient is an independent trial
- This simulates the stochastic nature of medical outcomes (same risk factors don't guarantee the same outcome)

**Example**:
- Computed risk = 0.412 (41.2%)
- Random draw: `u = 0.35`
- Since `0.35 < 0.412`, outcome = 1 (Alveolar Osteitis occurred)

### Configuration Files

- **`configs/risk_stats.json`**: Risk calculation rules (base incidences, modifiers, interactions, surgery modifiers)
- **`configs/extraction_type_stats.json`**: Surgical decision rules (base priors, rule categories, effects)
- **`configs/extraction_binary_stats.json`**: Removal indication rules
- **`configs/generation_config.json`**: Dataset size, decision model parameters (temperature, noise)
- **`configs/client_profiles.json`**: Client-specific variations (prevalence shifts, score scales, missingness)
- **`configs/noise_config.json`**: Feature noise parameters

### Output

The generator produces a CSV/Excel file with:
- **26 features**: Patient characteristics (demographics, symptoms, anatomy, pathology, systemic)
- **4 target variables**: Binary risk outcomes (`Risk_AlveolarOsteitis`, `Risk_SecondaryInfection`, `Risk_NerveDysesthesia`, `Risk_Bleeding`)
- **Decision variables**: `Surgical_Extraction_Type`, scores, probabilities (excluded from ML training to prevent leakage)
- **Metadata**: Client ID, Patient ID

---

## Machine Learning Pipeline

### Folder Structure

```
src/digit_fr/ml/
├── centralized/          # Centralized training (all data in one place)
│   └── train.py         # Main training script
├── federated/            # Federated learning (Flower framework)
├── local/                # Local training (per-client models)
├── config/              # Experiment configuration
│   └── experiment_config.py  # ExperimentConfig dataclass
├── constants.py          # Risk names and constants
├── data/                 # Data loading and preprocessing
│   ├── loaders.py       # Data loading with train/val/test splits
│   └── datasets.py      # PyTorch Dataset classes
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
    └── seed.py          # Seed management for reproducibility
```

### Model Architecture

#### MLP (Multi-Layer Perceptron)

**Location**: `src/digit_fr/ml/models/architectures/mlp.py`

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

### Training Pipeline

#### Step 1: Data Loading (`src/digit_fr/ml/data/loaders.py`)

1. **Load Raw Data**: Read CSV file
2. **Feature/Target Separation**:
   - **Features (X)**: All columns except targets and leakage variables
   - **Leakage Prevention**: Excludes `Patient`, `Client`, `Removal_Prob`, `Score_*`, `Prob_*` (these contain information about the decision process, not patient characteristics)
   - **Targets (y)**: 4 binary risk columns
3. **Missing Data Indicators**: Creates binary indicators for missing values in `Tooth_Mobility` and `Bone_Density`
4. **Train/Val/Test Split**: 
   - Default: 60% train, 20% val, 20% test
   - Uses `sklearn.model_selection.train_test_split` with random seed
5. **Categorical Encoding**: One-hot encoding for `Surgical_Extraction_Type` and `Tooth_Angulation`
6. **Imputation**: Median imputation for missing values (using `SimpleImputer`)

#### Step 2: Model Initialization

**Location**: `src/digit_fr/ml/centralized/train.py` (lines 55-60)

- Creates MLP with configurable architecture
- Input size determined from preprocessed data
- 4 classification heads (one per risk type)

#### Step 3: Loss Function and Optimization

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

#### Step 4: Training Loop (`src/digit_fr/ml/models/base/trainer.py`)

**BaseTrainer.fit()**:
1. **Forward Pass**: Model predicts logits for 4 risks
2. **Loss Calculation**: BCEWithLogitsLoss on predictions vs. true labels
3. **Backward Pass**: Gradient computation and optimizer step
4. **Validation**: After each epoch, evaluate on validation set
5. **History Tracking**: Stores train/val loss per epoch
6. **Checkpointing**: Saves model state after training

#### Step 5: Threshold Optimization

**Problem**: Binary classification with imbalanced classes requires optimal thresholds (not always 0.5)

**Methods** (`src/digit_fr/ml/metrics/threshold.py`):

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

#### Step 6: Evaluation (`src/digit_fr/ml/models/base/trainer.py`)

**BaseTrainer.evaluate()**:
1. **Inference**: Model in eval mode, no gradients
2. **Probability Conversion**: Apply sigmoid to logits → probabilities
3. **Binary Predictions**: Apply optimized thresholds to probabilities
4. **Metric Calculation** (`src/digit_fr/ml/metrics/calc_metrics.py`):
   - Per-risk metrics: Precision, Recall, F1, AUC-ROC, Specificity
   - Overall metrics: Macro-averaged and micro-averaged F1
   - Loss: Classification loss on test set

#### Step 7: Logging and Reporting

**WandB Integration** (`src/digit_fr/ml/metrics/report.py`):
- Logs experiment configuration (hyperparameters, data version, code version)
- Logs training history (loss per epoch)
- Logs dataset statistics (class distributions, imbalance ratios)
- Logs optimized thresholds
- Logs final test metrics

### Experiment Configuration

**Location**: `src/digit_fr/ml/config/experiment_config.py`

**ExperimentConfig** dataclass contains:
- **Experiment Type**: `centralized`, `local`, or `federated`
- **Seeds**: Separate seeds for data splitting and model initialization
- **Data Splits**: Train/val/test sizes, split strategy
- **Model**: Architecture type, hidden sizes, dropout
- **Training**: Batch size, learning rate, epochs, optimizer, scheduler
- **Loss**: Loss function, class weights
- **Evaluation**: Threshold optimization method
- **Versioning**: Data version (MD5 hash), code version (git commit)

### Key Design Patterns

1. **Base Classes**: `BaseModel` and `BaseTrainer` provide common interface for different architectures and training modes
2. **Modular Metrics**: Separate modules for calculation, threshold optimization, and reporting
3. **Reproducibility**: Comprehensive seed management and version tracking
4. **Leakage Prevention**: Careful exclusion of decision-related features from training data
5. **Multi-task Learning**: Single model predicts all 4 risks simultaneously

---

## Project Structure

### Root Directory

```
DIGIT-Federated-Recommenders/
├── configs/              # Configuration files (JSON)
│   ├── risk_stats.json           # Risk calculation rules
│   ├── extraction_type_stats.json  # Surgical decision rules
│   ├── extraction_binary_stats.json  # Removal indication rules
│   ├── generation_config.json    # Dataset generation parameters
│   ├── client_profiles.json      # Client-specific variations
│   └── noise_config.json         # Feature noise parameters
├── data/
│   └── raw/              # Generated datasets (CSV, Excel)
├── notebooks/            # Jupyter notebooks (EDA, testing)
├── src/digit_fr/
│   ├── core/             # Core utilities (paths, etc.)
│   ├── data_generation/  # Dataset generation system
│   │   ├── cli/          # Command-line interface
│   │   ├── config/       # Config loading
│   │   ├── generation/   # Main generation logic
│   │   └── rules/        # Decision rules (extraction, risk, removal)
│   └── ml/               # Machine learning pipeline
│       ├── centralized/  # Centralized training
│       ├── federated/    # Federated learning
│       ├── local/        # Local training
│       ├── config/       # Experiment configuration
│       ├── data/         # Data loading/preprocessing
│       ├── metrics/     # Evaluation metrics
│       ├── models/       # Model architectures and training
│       └── util/         # Utilities (seeding, etc.)
├── checkpoints/          # Saved model checkpoints
├── wandb/                # WandB experiment logs
└── pyproject.toml        # Python package configuration
```

---

## Summary

This project implements a **synthetic data generation system** for federated learning in the medical domain (wisdom tooth extraction complications). The generator uses evidence-based medical rules to simulate realistic patient data, clinical decisions, and stochastic risk outcomes. The **machine learning pipeline** trains multi-task neural networks to predict four binary risk outcomes, with careful attention to class imbalance, threshold optimization, and preventing data leakage.

The system is designed to support **federated learning** research, where multiple dental clinics (clients) collaborate to train models without sharing raw patient data, while accounting for realistic heterogeneity in data quality, clinical practices, and patient populations across clinics.