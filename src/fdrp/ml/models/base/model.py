from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the fdrp package.
    """

    def __init__(self, input_size: int, n_classification_classes: Optional[int] = None):
        super().__init__()
        self.n_features = input_size
        self.input_size = input_size
        self.n_classification_classes = n_classification_classes
    
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
                result["classification"] = (torch.sigmoid(outputs["classification"]) > 0.5).float()

            return result
        
    def get_config(self) -> Dict[str, Any]:
        config = {
            "n_features": self.n_features,
            "n_classification_classes": self.n_classification_classes,
        }
        return config