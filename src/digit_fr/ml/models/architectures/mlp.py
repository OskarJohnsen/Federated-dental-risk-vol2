from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from digit_fr.ml.models.base.model import BaseModel


class MLP(BaseModel):
    """
    MLP
    """
    def __init__(self, input_size: int, hidden_size: List[int] = [128, 64], dropout: float = 0.2, n_clf_classes: Optional[int] = None, n_reg_targets: Optional[int] = None):
        super().__init__(input_size, n_classification_classes=n_clf_classes, n_regression_targets=n_reg_targets)
        
        self.hidden_sizes = hidden_size
        self.dropout = dropout

        layers = []
        input_dim = input_size

        for hidden_dim in hidden_size:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)

        self.classification_head = nn.Linear(hidden_size[-1], n_clf_classes)
        self.regression_head = nn.Linear(hidden_size[-1], n_reg_targets)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}

        outputs["classification"] = self.classification_head(self.shared_layers(x))
        outputs["regression"] = torch.sigmoid(self.regression_head(self.shared_layers(x)))

        return outputs
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "architecture": "MLP"
        })
        return config