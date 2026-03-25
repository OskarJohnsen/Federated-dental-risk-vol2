from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from fdrp.ml.models.base.model import BaseModel

class MLP(BaseModel):
    """
    MLP
    """
    def __init__(self, input_size: int, hidden_size: List[int] = [128, 64], dropout: float = 0.2, n_clf_classes: Optional[int] = None):
        super().__init__(input_size, n_classification_classes=n_clf_classes)
        
        self.hidden_sizes = hidden_size
        self.dropout = dropout

        layers = []
        input_dim = input_size

        for hidden_dim in hidden_size:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)

        if n_clf_classes is not None:
            self.classification_heads = nn.ModuleList([
                nn.Linear(hidden_size[-1], 1) for _ in range(n_clf_classes)
            ])
        else:
            self.classification_heads = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        shared_features = self.shared_layers(x)
        if self.classification_heads is not None:
            risk_logits = []
            for head in self.classification_heads:
                risk_logits.append(head(shared_features))
            outputs["classification"] = torch.cat(risk_logits, dim=1)
        else:
            outputs["classification"] = torch.zeros(shared_features.shape[0], 0, device=shared_features.device)
        return outputs
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "architecture": "MLP"
        })
        return config