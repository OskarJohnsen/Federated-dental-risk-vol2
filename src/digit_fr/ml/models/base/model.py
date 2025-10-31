from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the digit_fr package.
    """

    def __init__(self, n_features: int, n_classification_classes: Optional[int] = None, n_regression_targets: Optional[int] = None):
        super().__init__()
        self.n_features = n_features
        self.n_classification_classes = n_classification_classes
        self.n_regression_targets = n_regression_targets
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # defined in mlp.py or other architectures
        pass

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            result = {}

            if "classification" in outputs:
                result["classification"] = torch.softmax(outputs["classification"], dim = -1)
            if "regression" in outputs:
                result["regression"] = torch.clamp(outputs["regression"], 0.0, 1.0)

            return result
        
    def get_config(self) -> Dict[str, Any]:
        config = {
            "n_features": self.n_features,
            "n_classification_classes": self.n_classification_classes,
            "n_regression_targets": self.n_regression_targets,
        }
        return config