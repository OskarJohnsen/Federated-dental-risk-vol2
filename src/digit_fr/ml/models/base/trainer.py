from typing import Any, Dict, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc
from sklearn.metrics import matthews_corrcoef
from ....core.paths import root_path, ensure_dir
from datetime import datetime

class BaseTrainer:
    """
    Base trainer for all training approaches (centralized, local, federated).
    """

    def __init__(self, model, device="cpu", seed = None, optimizer = None, loss_clf = None, scheduler = None, experiment_type: str = "base"):
        self.model = model
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.loss_clf = loss_clf
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
                
                loss_clf = 0.0
                
                if "classification" in outputs and "classification" in labels:
                    clf_labels = labels["classification"].to(self.device)
                    loss_clf = self.loss_clf(outputs["classification"], clf_labels)
                
                total_loss = loss_clf
                
                total_loss.backward()
                self.optimizer.step()
                running_loss += total_loss.item()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            avg_epoch_loss = running_loss / len(train_loader)
            self.training_history["train_loss"].append(avg_epoch_loss)
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics.get("loss_clf", 0.0)
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
        n_samples = 0
        

        all_clf_preds = []
        all_clf_labels = []
        all_clf_probs = []
        
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data[0].to(self.device), data[1]
                outputs = self.model(inputs)
                
                n_samples += len(inputs)
                
                if "classification" in outputs and "classification" in labels:
                    clf_logits = outputs["classification"]
                    clf_true = labels["classification"].to(self.device)
                    
                    clf_probs = torch.sigmoid(clf_logits)
                    clf_preds = (clf_probs > 0.5).float()
                    
                    all_clf_preds.append(clf_preds)
                    all_clf_labels.append(clf_true)
                    all_clf_probs.append(clf_probs)
                    
                    batch_loss_clf = self.loss_clf(clf_logits, clf_true)
                    total_loss_clf += batch_loss_clf.item() * len(inputs)
        
        # Concat all batches
        metrics = {}
        
        if all_clf_preds:
            clf_preds = torch.cat(all_clf_preds).int()
            clf_labels = torch.cat(all_clf_labels)
            clf_probs = torch.cat(all_clf_probs)
            
            metrics["loss_clf"] = total_loss_clf / n_samples
            risk_names = ["AlveolarOsteitis", "SecondaryInfection", "NerveDysesthesia", "Bleeding"]
            n_targets = clf_preds.shape[1]
            
            for i in range(n_targets):
                risk_pred = clf_preds[:, i]
                risk_label = clf_labels[:, i].int()
                risk_prob = clf_probs[:, i]
                
                metrics[f"accuracy_risk_{risk_names[i]}"] = accuracy(risk_pred, risk_label, task="binary")
                metrics[f"precision_risk_{risk_names[i]}"] = precision(risk_pred, risk_label, task="binary")
                metrics[f"recall_risk_{risk_names[i]}"] = recall(risk_pred, risk_label, task="binary")
                metrics[f"f1_risk_{risk_names[i]}"] = f1_score(risk_pred, risk_label, task="binary")
                metrics[f"roc_auc_risk_{risk_names[i]}"] = auroc(risk_prob, risk_label, task="binary")
                metrics[f"mcc_risk_{risk_names[i]}"] = matthews_corrcoef(risk_label.cpu().numpy(), risk_pred.cpu().numpy())
            
            # avg across all risks
            metrics["accuracy_clf_macro"] = sum([metrics[f"accuracy_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
            metrics["precision_clf_macro"] = sum([metrics[f"precision_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
            metrics["recall_clf_macro"] = sum([metrics[f"recall_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
            metrics["f1_clf_macro"] = sum([metrics[f"f1_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
            metrics["roc_auc_clf_macro"] = sum([metrics[f"roc_auc_risk_{risk_names[i]}"] for i in range(n_targets)]) / n_targets
        
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