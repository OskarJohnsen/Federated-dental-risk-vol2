# Setup Guide

This guide explains how to install and set up the project.

## Installation

Start by opening a terminal and navigating to the directory where you want the repository to be located.

Clone the repository:

```bash
git clone https://github.com/OskarJohnsen/Federated-dental-risk-vol2
cd federated-dental-risk-vol2
```

We recommend creating a virtual environment before installing the package:

```bash
python -m venv venv
```

Activate the environment:

```bash
# Mac/Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\activate
```

Install the package in editable mode:

```bash
pip install -e .
```

Installing in editable mode means that changes made to the source code automatically take effect without reinstalling the package.

---

## WandB Setup

Jonas/Smoothy, who previously worked on the project, used Weights & Biases (WandB) for experiment tracking and logging.

WandB is therefore integrated into parts of the training pipeline, but it is not strictly required for running experiments locally.

If you want to use WandB logging, start by logging into your account:

```bash
wandb login
```

This will open a browser window where you can authenticate your account.

You can optionally define a project name:

```bash
export WANDB_PROJECT=federated-dental-risk-prediction
```

You may replace `federated-dental-risk-prediction` with your own project name if desired.

If no project name is specified, the default defined in the codebase will be used.

You can also optionally specify your WandB username/entity:

```bash
export WANDB_ENTITY=your-username
```

---

## Disable WandB

If you do not want to use WandB, you can disable it completely by running:

```bash
export WANDB_MODE=disabled
```

This prevents experiments from being uploaded while still allowing all training scripts to run normally.

---

