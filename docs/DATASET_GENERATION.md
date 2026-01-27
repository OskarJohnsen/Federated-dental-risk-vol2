# Dataset Generation Guide

This document provides a comprehensive guide to the synthetic dataset generation system.

## Overview

The dataset generator creates synthetic patient records across multiple dental clinics (clients) in a federated learning setup. The generation process simulates realistic clinical decision-making and risk outcomes based on evidence-based medical rules.

## Quick Start

```bash
# Generate dataset with default settings
digit-fr-generate

# Generate with custom seed
digit-fr-generate --seed 42

# Generate without test set
digit-fr-generate --no-create-test-set

# Generate with custom output directory
digit-fr-generate --output-dir /path/to/output
```

## Medical Domain

### Context: Wisdom Tooth Extraction

This project simulates a **federated learning** scenario for predicting post-operative complications following wisdom tooth (third molar) extraction. The system generates synthetic patient data across multiple dental clinics (clients) and trains machine learning models to predict four types of surgical risks.

### Risk Types (Target Variables)

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

## Generation Pipeline

### Step 1: Patient Feature Generation

For each client (clinic), the system generates patient features using probabilistic models:

1. **Demographics**: Age (normal distribution, μ=28, σ=7, clipped 16-60), Sex (50/50), Mandibular/Maxillary (48/52)
2. **Symptoms**: Binary features with fixed probabilities (Pain: 50%, Swelling: 30%, Trismus: 20%, Pericoronitis: 40%)
3. **Anatomical Features**: Categorical distributions based on clinical prevalence
4. **Systemic Factors**: 
   - Conditional probabilities (e.g., osteoporosis depends on age and sex)
   - Bisphosphonates depend on osteoporosis status and age

### Step 2: Client-Specific Variations

Each client can have:
- **Prevalence Shifts**: Different age distributions, nerve proximity rates, impaction depth distributions
- **Score Scaling**: Multipliers for surgical decision scores (simulating different clinical preferences)
- **Missingness**: Missing data rates for specific features (simulating data quality heterogeneity)
- **Feature Noise**: Measurement noise applied to features (simulating inter-clinician variability)

### Step 3: Surgical Decision Generation

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

### Step 4: Risk Percentage Calculation (Target)

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

**Example Calculation** (Alveolar Osteitis):
```
base_incidence = 0.02
Impaction_Depth = 1 → multiplier = 1.0
Pericoronitis = 1 → multiplier = 1.4
Age = 30 → multiplier = 1.8 (age range 25-34)
Surgical_Extraction_Type = 1 → multiplier = 0.70

risk = 0.02 × 1.0 × 1.4 × 1.8 × 0.70 = 0.03528 (3.52%)
```

### Step 5: Bernoulli Draw (Binary Outcome Generation)

After computing the risk probability, the system performs a **Bernoulli trial** to determine the binary outcome:

**Process** (`src/digit_fr/data_generation/generation/synth.py`):
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

### Step 6: Risk Category Generation (Key Labels)

The **key labels** for evaluation are **risk categories**, not the binary outcomes. Categories are computed using **percentiles** on the risk probabilities:

**Process** (`src/digit_fr/data_generation/generation/synth.py`, lines 177-212):

1. **Percentile Calculation**: For each risk type, compute 33rd and 67th percentiles of risk probabilities
   - `p33 = np.percentile(prob_values, 33)`
   - `p67 = np.percentile(prob_values, 67)`

2. **Category Assignment**:
   - **Low (0)**: Risk probability < 33rd percentile
   - **Medium (1)**: 33rd percentile ≤ Risk probability < 67th percentile
   - **High (2)**: Risk probability ≥ 67th percentile

3. **Composite Category**: For each patient, compute `Risk_Category_Composite` as the maximum severity across all four risks (this is only for the label skew partitioning)

**Why Categories Instead of Binary Outcomes?**
- The Bernoulli draw introduces significant noise, especially for low-probability risks (e.g., Bleeding: 0.08% base incidence)
- Models can learn the dataset distribution but struggle to predict the exact small percentages that get Bernoulli-drawn
- Categories provide a more stable and clinically meaningful evaluation target
- Categories represent risk severity levels, which are more actionable for clinical decision-making

**Output Columns**:
- `Risk_Category_AlveolarOsteitis`: 0 (low), 1 (medium), or 2 (high)
- `Risk_Category_SecondaryInfection`: 0, 1, or 2
- `Risk_Category_NerveDysesthesia`: 0, 1, or 2
- `Risk_Category_Bleeding`: 0, 1, or 2
- `Risk_Category_Composite`: Maximum severity across all risks (0, 1, or 2)

**Global Thresholds**: The percentile boundaries (33rd and 67th) are saved to `configs/global_thresholds/{DATASET}/global_thresholds_{IID_TYPE}.json` for use during model evaluation.

## IID/Non-IID Partitioning

The system uses **NIID-Bench** partitioning (Dirichlet distribution) to create IID or non-IID data distributions across clients.

### IID Partitioning
- Data is randomly shuffled and evenly distributed across clients
- Each client has similar label distributions

### Non-IID Partitioning
- Uses Dirichlet distribution with concentration parameter `beta` (beta_L for label skew, beta_Q for quantity skew)
- Creates heterogeneous distributions:
  - **Label Skew**: Different clients have different label distributions
  - **Quantity Skew**: Different clients have different amounts of data

**Code Location**: `src/digit_fr/data_generation/partitioning/niid_bench_partitioning.py`

## Configuration Files

- **`configs/risk_stats.json`**: Risk calculation rules (base incidences, modifiers, interactions, surgery modifiers)
- **`configs/extraction_type_stats.json`**: Surgical decision rules (base priors, rule categories, effects)
- **`configs/extraction_binary_stats.json`**: Removal indication rules
- **`configs/generation_config.json`**: Dataset size, decision model parameters (temperature, noise)
- **`configs/client_profiles.json`**: Client-specific variations (prevalence shifts, score scales, missingness)
- **`configs/noise_config.json`**: Feature noise parameters

## Output

The generator produces:

### CSV/Excel File
- **26 features**: Patient characteristics (demographics, symptoms, anatomy, pathology, systemic)
- **4 binary risk outcomes**: `Risk_AlveolarOsteitis`, `Risk_SecondaryInfection`, `Risk_NerveDysesthesia`, `Risk_Bleeding` (used for training)
- **4 risk probability columns**: `Risk_*_Prob` (true risk probabilities, used for evaluation)
- **5 risk category columns**: `Risk_Category_*` (0=low, 1=medium, 2=high) - **these are the key labels for evaluation**
- **Decision variables**: `Surgical_Extraction_Type`, scores, probabilities (excluded from ML training to prevent leakage)
- **Metadata**: Client ID, Patient ID

### Global Test Set
- Automatically created when `--create-test-set` is enabled (default: True)
- Saved to `data/processed/{DATASET}/global_test_set_{IID_TYPE}.csv`
- Configurable size (default: 3000 samples)
- Original dataset is backed up (default: True)

### Global Thresholds
- Saved to `configs/global_thresholds/{DATASET}/global_thresholds_{IID_TYPE}.json`
- Contains partition metadata (beta values, heterogeneity metrics)
- Used for WandB logging and analysis

## CLI Options

```bash
digit-fr-generate [OPTIONS]

Options:
  --seed INT              Random seed override (default: from config)
  --output-dir PATH       Output directory (default: from config or data/raw/)
  --formats TEXT          Comma-separated formats: csv,xlsx (default: csv,xlsx)
  --create-test-set       Create global test set automatically (default: True)
  --test-samples INT      Number of samples in test set (default: 3000)
  --test-seed INT         Random seed for test set splitting (default: 999)
  --backup                Create backup of original dataset (default: True)
```

## Code Structure

```
src/digit_fr/data_generation/
├── cli/
│   └── generate.py          # CLI entry point
├── config/
│   └── loader.py            # Configuration loading
├── generation/
│   └── synth.py             # Main generation logic
├── partitioning/
│   └── niid_bench_partitioning.py  # IID/non-IID partitioning
├── profiles/
│   └── generator.py         # Client profile generation
├── rules/
│   ├── decision/
│   │   ├── extraction.py    # Surgical decision rules
│   │   ├── risk.py          # Risk calculation rules
│   │   └── removal.py       # Removal indication rules
│   └── noise/
│       └── apply.py         # Feature noise application
└── splits.py                # Test set creation
```

## Examples

### Basic Generation
```bash
# Generate dataset with defaults
digit-fr-generate
```

### Custom Seed
```bash
# Reproducible generation
digit-fr-generate --seed 42
```

### Multiple Formats
```bash
# Generate both CSV and Excel
digit-fr-generate --formats csv,xlsx
```

### Without Test Set
```bash
# Skip test set creation
digit-fr-generate --no-create-test-set
```

## Troubleshooting

### Import Errors
- Ensure package is installed: `pip install -e .`
- Check Python version: `python --version` (should be >= 3.10)

### Configuration Errors
- Verify all config files exist in `configs/`
- Check JSON syntax is valid
- Ensure required keys are present in configs

### Path Issues
- Default output directory is `data/raw/` relative to project root
- Use `--output-dir` to specify custom location
- Ensure write permissions for output directory