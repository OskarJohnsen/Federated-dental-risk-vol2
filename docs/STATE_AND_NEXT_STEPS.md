# Project State & Next Steps

## Current Project State

The project evaluates centralized, local, and federated learning on synthetic dental risk data. The system is fully operational with the following capabilities:

### Dataset Generation
- **IID and non-IID data partitioning** using NIID-Bench (Dirichlet distribution)
- **Label skew** (β_L): Controls heterogeneity of label distributions across clients
- **Quantity skew** (β_Q): Controls heterogeneity of data amounts across clients
- **Composite label skew** across 4 risk categories (Alveolar Osteitis, Secondary Infection, Nerve Dysesthesia, Bleeding)
- **Risk categorization**: Converts risk probabilities to Low/Medium/High categories using 33rd/67th percentile thresholds
- **Automatic test set creation** with configurable size and seed

### Machine Learning Pipeline
- **Three training paradigms**: Centralized, Local (per-client), Federated Learning
- **Multi-task learning**: Single model predicts all 4 binary risk outcomes simultaneously
- **Dual evaluation approach**:
  - **Probability metrics**: MSE/MAE comparing predicted vs. true risk probabilities
  - **Category metrics**: F1/Accuracy comparing predicted vs. true risk categories
- **Threshold strategies**: 
  - **Global thresholds**: Percentile boundaries from dataset generation (consistent across clients)
  - **Per-client thresholds**: Computed from each client's validation set (accounts for client-specific distributions)
- **WandB integration**: Comprehensive experiment tracking and logging

### Current Configuration (**for non-iid datasets**)
- **Default label skew β**: 0.5 (moderate heterogeneity)
- **Default quantity skew β**: 0.7 (moderate quantity imbalance)
- **10 clients** with 5,000 patients per client (50,000 total)
- **4 risk types** with varying base incidences (0.08% to 2%)

---

## What was the last thing tested

The most recent experiments explored how model performance changes with the Dirichlet β parameter controlling label and quantity skew.

### Experimental Setup
- **6 different β values** were tested (from 0.1 to 10)
- **Label skew β = Quantity skew β** (no isolation between effects)
- **Experiments were run manually** (no automated sweep script)
- **Results location**: `data/results/A/non-iid/sweep_beta` (contains summary PDFs and CSV exports)

### What This Means
- This experiment is **partially done** — initial exploration completed
- **No systematic sweep** has been implemented yet
- **Effects are confounded** — label skew and quantity skew were varied together, making it impossible to isolate their individual impacts

---

## What is still unknown

The following critical research questions remain unanswered:

### Isolated Effects
1. **Impact of label skew alone**: How does varying β_L (with β_Q held constant at high value ≈ IID) affect model performance?
2. **Impact of quantity skew alone**: How does varying β_Q (with β_L held constant at high value ≈ IID) affect model performance?
3. **Interaction effects**: How do label skew and quantity skew interact? Do they compound or mitigate each other's effects?

### Threshold Strategy Interactions
4. **Threshold strategy performance**: How does the choice between global vs. per-client thresholds interact with different levels of heterogeneity?
   - Does per-client thresholding help more under high label skew?
   - Does global thresholding remain stable across quantity skew levels?

### Performance Metrics
5. **Metric sensitivity**: Which metrics (F1, MSE, Fleiss κ, ECE) are most sensitive to different types of skew?
6. **Training paradigm comparison**: How do centralized, local, and federated learning respond differently to isolated label vs. quantity skew?

### Parameter Space
7. **Systematic β exploration**: What is the full β-parameter space that should be explored?

---

## What should / could be done next

### Next Steps

#### 1. Implement Automated Sweep Script
Create a script that systematically varies β parameters and runs experiments:
- **Input**: Grid of β_L and β_Q values
- **Output**: Organized results directory with metadata
- **Features**:
  - Generate datasets with different β combinations (+ store their specific thresholds and global_test_set)
  - Run training for all three paradigms (centralized, local, federated)
  - Export results to structured format
  - Track experiment metadata (β values, seeds, timestamps)

#### 2. Isolate Label Skew Effects
**Experiment**: Vary label skew β_L while keeping quantity skew β_Q high (≈ IID)
- **β_L values**: e.g. from [0.1 - 2.0]
- **β_Q**: Fixed at 10.0 (essentially IID for quantity)
- **Measure**:
  - F1 global and per-client
  - MSE (probability prediction)
  - Fleiss κ (inter-rater agreement)
  - ECE (calibration error)
- **Compare**: Performance across centralized, local, and federated paradigms

#### 3. Isolate Quantity Skew Effects
**Experiment**: Vary quantity skew β_Q while keeping label skew β_L high (≈ IID)
- **β_Q values**: e.g. from [0.1 - 2.0]
- **β_L**: Fixed at 10.0 (essentially IID for labels)
- **Measure**: Same metrics as above
- **Observe**: How quantity imbalance alone affects model performance

#### 4. Interaction Analysis
**Experiment**: Full factorial design with both β_L and β_Q
- **Grid**: All combinations from steps 2 and 3
- **Focus**: Identify interaction patterns
- **Visualize**: Heatmaps showing metric values across β_L × β_Q space

#### 5. Threshold Strategy Comparison
For each β combination, compare:
- **Global thresholds**: Performance using dataset-wide percentile boundaries
- **Per-client thresholds**: Performance using client-specific percentile boundaries
- **Question**: Which strategy is more robust under different heterogeneity levels?

---

## Findings so far

### Current Results (No Isolation)
Results from experiments with **no isolation** (β_L = β_Q) using 6 different beta values:
- **Location**: `data/results/A/non-iid/sweep_beta/`
- **Files**: `summary.pdf`, `wandb_export_all_metrics.csv`, and individual metric PDFs
- **Note**: These results are **confounded** — cannot distinguish label skew effects from quantity skew effects

### Configuration Notes
- **Current default**: β_L = 0.5, β_Q = 0.7 (moderate heterogeneity)
- **Partition metadata** is automatically saved in `configs/global_thresholds/{DATASET}/global_thresholds_{IID_TYPE}.json`
- **Heterogeneity metrics** are computed and logged (entropy, diversity, imbalance ratios, Gini coefficient)

*Last updated: February 2026*
*Document version: 1.0*