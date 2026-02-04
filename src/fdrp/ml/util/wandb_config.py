"""
WandB configuration utilities.

Allows each user to configure their own WandB project via environment variables.
Defaults to sensible values if not set.
"""
import os
from typing import Optional


def get_wandb_project() -> str:
    """
    Get WandB project name from environment variable.
    
    Defaults to 'federated-dental-risk-prediction' if not set.
    
    Users can override by setting WANDB_PROJECT environment variable:
        export WANDB_PROJECT=my-project-name
    
    Returns:
        WandB project name string
    """
    return os.getenv("WANDB_PROJECT", "federated-dental-risk-prediction")


def get_wandb_entity() -> Optional[str]:
    """
    Get WandB entity (user/team) from environment variable.
    
    Defaults to None, which means WandB will use the logged-in user's account.
    
    Users can override by setting WANDB_ENTITY environment variable:
        export WANDB_ENTITY=my-username
    
    Returns:
        WandB entity name string, or None to use logged-in user
    """
    return os.getenv("WANDB_ENTITY", None)
