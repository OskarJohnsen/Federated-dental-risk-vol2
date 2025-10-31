from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers (centralized, local, federated).
    """

    def __init__(self, model, device="cpu", seed=None, optimizer=None, loss_clf=None, loss_reg=None):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.loss_clf = loss_clf
        self.loss_reg = loss_reg
        self.model.to(self.device)
        self.training_history = {"train_loss": [], "val_loss": []}
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def fit(self, train_loader, val_loader=None, epochs: int = 10):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0 
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss_clf = 0.0
                loss_reg = 0.0
                
                if "classification" in outputs and "classification" in labels:
                    loss_clf = self.loss_clf(outputs["classification"], labels["classification"])
                
                if "regression" in outputs and "regression" in labels:
                    loss_reg = self.loss_reg(outputs["regression"], labels["regression"])
                
                total_loss = loss_clf + loss_reg
                
                total_loss.backward()
                self.optimizer.step()
                running_loss += total_loss.item()
            
            avg_epoch_loss = running_loss / len(train_loader)
            self.training_history["train_loss"].append(avg_epoch_loss)
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics.get("loss", 0.0)
                self.training_history["val_loss"].append(val_loss)
                print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss}, Val Loss: {val_loss}')
            else:
                print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss}')
        
        return {
            "history": self.training_history,
            "final_train_loss": self.training_history["train_loss"][-1],
            "final_val_loss": self.training_history["val_loss"][-1] if val_loader else None,
        }
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(inputs)
                
                loss_clf = 0.0
                loss_reg = 0.0
                
                if "classification" in outputs and "classification" in labels:
                    loss_clf = self.loss_clf(outputs["classification"], labels["classification"])
                
                if "regression" in outputs and "regression" in labels:
                    loss_reg = self.loss_reg(outputs["regression"], labels["regression"])
                
                total_loss += (loss_clf + loss_reg) * len(inputs)
                n_samples += len(inputs)
        
        if n_samples > 0:
            avg_loss = total_loss / n_samples
        else:
            avg_loss = 0.0
        
        return {
            "loss": avg_loss,
            # TODO: Add other metrics (accuracy, MSE, etc.)
        }
    
    def save_checkpoint(self):
        # TODO
        pass

    def load_checkpoint(self):
        # TODO
        pass