from typing import Any, Dict, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc, r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import matthews_corrcoef
from ....core.paths import root_path, ensure_dir
from datetime import datetime


class BaseTrainer:
    """
    Base trainer for all training approaches (centralized, local, federated).
    """

    def __init__(self, model, device="cpu", seed = None, optimizer = None, loss_clf = None, loss_reg = None, scheduler = None, experiment_type: str = "base"):
        self.model = model
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.loss_clf = loss_clf
        self.loss_reg = loss_reg
        self.scheduler = scheduler
        self.model.to(self.device)
        self.training_history = {"train_loss": [], "val_loss": []}
        self.seed = seed
        self.experiment_type = experiment_type
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def fit(self, train_loader, val_loader=None, epochs: int = 10):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0 
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                lambda_clf = 1.0
                lambda_reg = 200.0
                loss_clf = 0.0
                loss_reg = 0.0
                
                if "classification" in outputs and "classification" in labels:
                    clf_labels = labels["classification"].to(self.device)
                    loss_clf = self.loss_clf(outputs["classification"], clf_labels)
                
                if "regression" in outputs and "regression" in labels:
                    reg_labels = labels["regression"].to(self.device)
                    loss_reg = self.loss_reg(outputs["regression"], reg_labels)
                
                total_loss = lambda_clf * loss_clf + lambda_reg * loss_reg
                
                total_loss.backward()
                self.optimizer.step()
                running_loss += total_loss.item()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            avg_epoch_loss = running_loss / len(train_loader)
            self.training_history["train_loss"].append(avg_epoch_loss)
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics.get("avg_loss", 0.0)
                self.training_history["val_loss"].append(val_loss)
                print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss}')
        
        checkpoint_path = self.save_checkpoint(experiment_type=self.experiment_type)
        
        return {
            "history": self.training_history,
            "final_train_loss": self.training_history["train_loss"][-1],
            "final_val_loss": self.training_history["val_loss"][-1] if val_loader else None,
            "checkpoint_path": checkpoint_path,
        }
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        self.model.eval()
        total_loss_clf = 0.0
        total_loss_reg = 0.0
        n_samples = 0
        

        all_clf_preds = []
        all_clf_labels = []
        all_clf_probs = []
        all_reg_preds = []
        all_reg_labels = []
        
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data[0].to(self.device), data[1]
                outputs = self.model(inputs)
                
                n_samples += len(inputs)
                
                if "classification" in outputs and "classification" in labels:
                    clf_logits = outputs["classification"]
                    clf_true = labels["classification"].to(self.device)
                    
                    all_clf_preds.append(clf_logits.argmax(dim=1))
                    all_clf_labels.append(clf_true)
                    all_clf_probs.append(torch.softmax(clf_logits, dim=1))
                    
                    batch_loss_clf = self.loss_clf(clf_logits, clf_true)
                    total_loss_clf += batch_loss_clf.item() * len(inputs)
                
                if "regression" in outputs and "regression" in labels:
                    reg_pred = outputs["regression"]
                    reg_true = labels["regression"].to(self.device)
                    
                    all_reg_preds.append(reg_pred)
                    all_reg_labels.append(reg_true)
                    
                    batch_loss_reg = self.loss_reg(reg_pred, reg_true)
                    total_loss_reg += batch_loss_reg.item() * len(inputs)
        
        # Concat all batches
        metrics = {}
        
        if all_clf_preds:
            clf_preds = torch.cat(all_clf_preds)
            clf_labels = torch.cat(all_clf_labels)
            clf_probs = torch.cat(all_clf_probs)
            num_classes = clf_probs.shape[1]
            
            metrics["loss_clf"] = total_loss_clf / n_samples
            metrics["accuracy_clf"] = accuracy(clf_preds, clf_labels, task="multiclass", num_classes=num_classes)
            metrics["precision_clf"] = precision(clf_preds, clf_labels, task="multiclass", num_classes=num_classes, average="macro")
            metrics["recall_clf"] = recall(clf_preds, clf_labels, task="multiclass", num_classes=num_classes, average="macro")
            metrics["f1_clf"] = f1_score(clf_preds, clf_labels, task="multiclass", num_classes=num_classes, average="macro")
            metrics["roc_auc_clf"] = auroc(clf_probs[:, 1], clf_labels, task="binary")
            metrics["mcc_clf"] = matthews_corrcoef(clf_labels.cpu().numpy(), clf_preds.cpu().numpy())
        
        if all_reg_preds:
            reg_preds = torch.cat(all_reg_preds)
            reg_labels = torch.cat(all_reg_labels)

            metrics["loss_reg"] = total_loss_reg / n_samples if n_samples > 0 else 0.0
            metrics["mse_reg"] = mean_squared_error(reg_preds, reg_labels)
            metrics["rmse_reg"] = torch.sqrt(metrics["mse_reg"])
            metrics["mae_reg"] = mean_absolute_error(reg_preds, reg_labels)
            
            # r2 per target
            n_targets = reg_preds.shape[1]
            r2_scores = [r2_score(reg_preds[:, i], reg_labels[:, i]) for i in range(n_targets)]
            metrics["r2_reg"] = torch.mean(torch.tensor(r2_scores))
            
            if torch.all(reg_preds >= 0) and torch.all(reg_labels >= 0):
                metrics["rmsle_reg"] = torch.sqrt(mean_squared_error(torch.log(1 + reg_preds), torch.log(1 + reg_labels)))
        

        metrics["avg_loss"] = metrics.get("loss_clf", 0.0) + metrics.get("loss_reg", 0.0)
        
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
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'epoch': len(self.training_history['train_loss']),
            'device': str(self.device),
            'seed': self.seed,
            'model_class': self.model.__class__.__name__,
            'optimizer_class': self.optimizer.__class__.__name__,
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

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_history and checkpoint['training_history']:
            self.training_history = checkpoint['training_history']

        if checkpoint.get('seed') is not None:
            self.seed = checkpoint['seed']
        
        print(f"Checkpoint loaded successfully at {checkpoint.get('epoch')}")
        
        return checkpoint