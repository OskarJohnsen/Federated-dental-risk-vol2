from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from fdrp.ml.models.base.model import BaseModel


class MLP(BaseModel):
    """
    MLP with shared trunk + task-specific heads

    head_hidden_size = 0  -> linear heads (old model)
    head_hidden_size > 0  -> nonlinear heads (new model)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: List[int] = [128, 64],
        head_hidden_size: int = 16,
        dropout: float = 0.2,
        n_clf_classes: Optional[int] = None
    ):
        super().__init__(input_size, n_classification_classes=n_clf_classes)
        
        self.hidden_sizes = hidden_size
        self.head_hidden_size = head_hidden_size
        self.dropout = dropout

        # -------------------------
        # Shared trunk
        # -------------------------
        layers = []
        input_dim = input_size

        for hidden_dim in hidden_size:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)

        # -------------------------
        # Task-specific heads
        # -------------------------
        if n_clf_classes is not None:
            self.classification_heads = nn.ModuleList([
                self._build_head(hidden_size[-1], head_hidden_size, dropout)
                for _ in range(n_clf_classes)
            ])
        else:
            self.classification_heads = None

    # -------------------------
    # Head builder (KEY CHANGE)
    # -------------------------
    def _build_head(self, input_dim: int, head_hidden_size: int, dropout: float):
        if head_hidden_size == 0:
            # OLD MODEL (linear head)
            return nn.Linear(input_dim, 1)
        else:
            # NEW MODEL (nonlinear head)
            return nn.Sequential(
                nn.Linear(input_dim, head_hidden_size),
                nn.BatchNorm1d(head_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_size, 1)
            )

    # -------------------------
    # Forward
    # -------------------------
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
    
    # -------------------------
    # Config
    # -------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_sizes": self.hidden_sizes,
            "head_hidden_size": self.head_hidden_size,
            "dropout": self.dropout,
            "architecture": "MLP_shared_trunk_task_specific_heads"
        })
        return config
        