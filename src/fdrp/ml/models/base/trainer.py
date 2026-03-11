from typing import Any, Dict, Optional
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from ...constants import RISK_NAMES
from ...metrics.calc_metrics import model_metrics, model_metrics_probability
from ....core.paths import root_path, ensure_dir
from datetime import datetime

class BaseTrainer:
    """
    Base trainer for all training approaches (centralized, local, federated).
    Supports optional FedProx via prox_mu > 0 and reference_params.
    """

    def __init__(
        self,
        model,
        device="cpu",
        seed=None,
        optimizer=None,
        loss_clf=None,
        scheduler=None,
        experiment_type: str = "base",
        prox_mu: float = 0.0,
        reference_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.loss_clf = loss_clf
        self.scheduler = scheduler
        self.model.to(self.device)
        self.training_history = {"train_loss": [], "val_loss": []}
        self.seed = seed
        self.experiment_type = experiment_type

        # FedProx
        self.prox_mu = prox_mu
        self.reference_params = reference_params

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _compute_fedprox_penalty(self) -> torch.Tensor:
        """
        Compute sum ||w - w_global||^2 over trainable parameters.
        Returns 0 if FedProx is disabled.
        """
        if self.prox_mu <= 0.0 or self.reference_params is None:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            ref_param = self.reference_params[name].to(self.device)
            penalty = penalty + torch.sum((param - ref_param) ** 2)

        return penalty

    def fit(self, train_loader, val_loader=None, epochs: int = 10):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            running_base_loss = 0.0
            running_prox_loss = 0.0

            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss_clf = torch.tensor(0.0, device=self.device)

                if "classification" in outputs and "classification" in labels:
                    clf_labels = labels["classification"].to(self.device)
                    loss_clf = self.loss_clf(outputs["classification"], clf_labels)

                prox_penalty = self._compute_fedprox_penalty()
                prox_loss = 0.5 * self.prox_mu * prox_penalty
                total_loss = loss_clf + prox_loss

                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()
                running_base_loss += loss_clf.item()
                running_prox_loss += prox_loss.item()

            if self.scheduler is not None:
                self.scheduler.step()

            avg_epoch_loss = running_loss / len(train_loader)
            avg_base_loss = running_base_loss / len(train_loader)
            avg_prox_loss = running_prox_loss / len(train_loader)

            self.training_history["train_loss"].append(avg_epoch_loss)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics.get("loss_clf", 0.0)
                self.training_history["val_loss"].append(val_loss)

                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_epoch_loss:.4f}, "
                    f"Base Loss: {avg_base_loss:.4f}, "
                    f"Prox Loss: {avg_prox_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_epoch_loss:.4f}, "
                    f"Base Loss: {avg_base_loss:.4f}, "
                    f"Prox Loss: {avg_prox_loss:.4f}"
                )

        checkpoint_path = self.save_checkpoint(experiment_type=self.experiment_type)

        return {
            "history": self.training_history,
            "final_train_loss": self.training_history["train_loss"][-1],
            "final_val_loss": self.training_history["val_loss"][-1] if val_loader else None,
            "checkpoint_path": checkpoint_path,
        }

    def evaluate(self, data_loader, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        self.model.eval()
        total_loss_clf = 0.0
        n_samples = 0

        all_clf_preds = []
        all_clf_labels = []
        all_clf_probs = []
        all_true_probs = []
        all_true_categories = []
        risk_names = RISK_NAMES

        if thresholds is None:
            thresholds = {name: 0.5 for name in risk_names}

        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data[0].to(self.device), data[1]
                outputs = self.model(inputs)

                n_samples += len(inputs)

                if "classification" in outputs and "classification" in labels:
                    clf_logits = outputs["classification"]
                    clf_true = labels["classification"].to(self.device)

                    clf_probs = torch.sigmoid(clf_logits)

                    clf_preds = torch.zeros_like(clf_probs)
                    for i, risk_name in enumerate(risk_names):
                        threshold = thresholds.get(risk_name, 0.5)
                        clf_preds[:, i] = (clf_probs[:, i] > threshold).float()

                    all_clf_preds.append(clf_preds)
                    all_clf_labels.append(clf_true)
                    all_clf_probs.append(clf_probs)

                    if "probabilities" in labels:
                        all_true_probs.append(labels["probabilities"].to(self.device))

                    if "categories" in labels:
                        all_true_categories.append(labels["categories"].to(self.device))

                    batch_loss_clf = self.loss_clf(clf_logits, clf_true)
                    total_loss_clf += batch_loss_clf.item() * len(inputs)

        metrics = {}

        if all_clf_preds:
            clf_preds = torch.cat(all_clf_preds).int()
            clf_labels = torch.cat(all_clf_labels)
            clf_probs = torch.cat(all_clf_probs)

            metrics["loss_clf"] = total_loss_clf / n_samples

            if len(all_true_probs) > 0:
                clf_true_probs = torch.cat(all_true_probs)
                prob_metrics = model_metrics_probability(clf_probs, clf_true_probs, risk_names)
                metrics.update(prob_metrics)
                metrics["_true_probs"] = clf_true_probs.cpu().numpy()

            if len(all_true_categories) > 0:
                clf_true_categories = torch.cat(all_true_categories)
                metrics["_true_categories"] = clf_true_categories.cpu().numpy()

            metrics["_probs"] = clf_probs.cpu().numpy()
            metrics["_labels"] = clf_labels.cpu().numpy()
            metrics["_risk_names"] = risk_names

        return metrics

    def save_checkpoint(self, filepath: Optional[Path] = None, include_history: bool = True, experiment_type: str = "base"):
        if filepath is None:
            model_name = self.model.__class__.__name__
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = root_path("checkpoints", experiment_type, f"{model_name}_{timestamp}.pt")
        else:
            filepath = Path(filepath)
        ensure_dir(filepath.parent)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'training_history': self.training_history,
            'epoch': len(self.training_history['train_loss']),
            'device': str(self.device),
            'seed': self.seed,
            'model_class': self.model.__class__.__name__,
            'optimizer_class': self.optimizer.__class__.__name__ if self.optimizer is not None else None,
            'timestamp': datetime.now().isoformat(),
            'model_config': self.model.get_config(),
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
        return filepath

    def load_checkpoint(self, filepath: Path, load_history: bool = True):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if load_history and checkpoint['training_history']:
            self.training_history = checkpoint['training_history']

        if checkpoint.get('seed') is not None:
            self.seed = checkpoint['seed']

        print(f"Checkpoint loaded successfully at {checkpoint.get('epoch')}")

        return checkpoint