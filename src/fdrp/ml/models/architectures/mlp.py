from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from fdrp.ml.models.base.model import BaseModel


class MLP(BaseModel):

    def __init__(self, input_size: int, hidden_size: List[int] = [64, 128,64], dropout: float = 0.2, n_clf_classes: Optional[int] = None):
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

"""

class MLPBlock(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()

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

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)
        logits = self.output_layer(features)
        return logits


class MLP(BaseModel):

    def __init__(
        self,
        input_size: int,
        hidden_size: List[int] = [128, 64],
        dropout: float = 0.2,
        n_clf_classes: Optional[int] = None
    ):
        super().__init__(input_size, n_classification_classes=n_clf_classes)

        self.hidden_sizes = hidden_size
        self.dropout = dropout

        if n_clf_classes is not None:
            self.classification_heads = nn.ModuleList([
                MLPBlock(input_size, hidden_size, dropout)
                for _ in range(n_clf_classes)
            ])
        else:
            self.classification_heads = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}

        if self.classification_heads is not None:
            risk_logits = []
            for net in self.classification_heads:
                risk_logits.append(net(x))
            outputs["classification"] = torch.cat(risk_logits, dim=1)
        else:
            outputs["classification"] = torch.zeros(x.shape[0], 0, device=x.device)

        return outputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "architecture": "MLP_separate_networks"
        })
        return config


class MLP(BaseModel):

    def __init__(
        self,
        input_size: int,
        hidden_size: List[int] = [128, 64],
        dropout: float = 0.2,
        n_clf_classes: Optional[int] = None
    ):
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
        shared_output_dim = input_dim

        # Hardcoded head size
        head_hidden_size = 32

        if n_clf_classes is not None:
            self.classification_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(shared_output_dim, head_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_size, 1)
                )
                for _ in range(n_clf_classes)
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
            outputs["classification"] = torch.zeros(
                shared_features.shape[0], 0, device=shared_features.device
            )

        return outputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "architecture": "MLP"
        })
        return config

# 2 hidden head lag:


class MLP(BaseModel):

    def __init__(
        self,
        input_size: int,
        hidden_size: List[int] = [64,64,64],
        dropout: float = 0.2,
        n_clf_classes: Optional[int] = None,
    ):
        super().__init__(input_size, n_classification_classes=n_clf_classes)

        self.hidden_sizes = hidden_size
        self.dropout = dropout

        # ===== Shared trunk =====
        layers = []
        input_dim = input_size

        for hidden_dim in hidden_size:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)
        shared_output_dim = input_dim

        # ===== HEAD CONFIG (HER styrer du det) =====
        head_hidden_sizes = [256*2, 256]  

        if n_clf_classes is not None:
            self.classification_heads = nn.ModuleList([
                self._build_head(shared_output_dim, head_hidden_sizes, dropout)
                for _ in range(n_clf_classes)
            ])
        else:
            self.classification_heads = None

    def _build_head(self, input_dim, hidden_sizes, dropout):
        layers = []
        current_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h

        layers.append(nn.Linear(current_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        shared_features = self.shared_layers(x)

        if self.classification_heads is not None:
            logits = [head(shared_features) for head in self.classification_heads]
            outputs["classification"] = torch.cat(logits, dim=1)
        else:
            outputs["classification"] = torch.zeros(
                shared_features.shape[0], 0, device=shared_features.device
            )

        return outputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_sizes": self.hidden_sizes,
            "architecture": "MLP_multi_head"
        })
        return config
"""