from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
import subprocess
import hashlib
from pathlib import Path

# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script#comment122171839_14989858
def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
    except:
        return "unknown"

# 
def get_data_version(data_path: str) -> str:
    try:
        with open(data_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except:
        return "unknown"

@dataclass
class ExperimentConfig:
    
    # experiment
    experiment_type: Literal["centralized", "local", "federated"]
    experiment_id: str
    run_name: Optional[str] = None
    
    # seeds
    data_split_seed: int = 42
    model_seed: int = 42
    
    # data
    test_size: float = 0.2
    val_size: float = 0.2
    split_strategy: Literal["random", "stratified", "client_aware"] = "random"
    
    # model
    model: Literal["MLP"] = "MLP"
    hidden_size: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    input_size: Optional[int] = None
    
    # parameter
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 15
    optimizer: Literal["Adam", "Adagrad", "AdamW"] = "Adam"
    
    # scheduler
    scheduler: Optional[Literal["StepLR"]] = "StepLR"
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5
    
    # loss
    loss_function: Literal["BCEWithLogitsLoss"] = "BCEWithLogitsLoss"
    use_class_weights: bool = False
    
    # eval
    threshold_method: Literal["f1", "youden"] = "youden"
    
    # versions
    data_version: Optional[str] = None
    code_version: Optional[str] = None
    
    # federated (mentioned in flower, maybe add / use later)
    federated_rounds: Optional[int] = None
    clients_per_round: Optional[int] = None
    local_epochs: Optional[int] = None
    aggregation_method: Optional[str] = None
    
    def __post_init__(self):
        if self.code_version is None:
            self.code_version = get_git_commit_hash()
    
    @property
    def train_size(self) -> float:
        return 1.0 - self.test_size - self.val_size
    
    def to_wandb_config(self) -> Dict[str, Any]:
        config = {
            "experiment_type": self.experiment_type,
            "experiment_id": self.experiment_id,
            "data_split_seed": self.data_split_seed,
            "model_seed": self.model_seed,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "train_size": self.train_size,
            "split_strategy": self.split_strategy,
            "model": self.model,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
            "loss_function": self.loss_function,
            "use_class_weights": self.use_class_weights,
            "threshold_method": self.threshold_method,
        }
        
        if self.input_size is not None:
            config["input_size"] = self.input_size
        
        if self.data_version:
            config["data_version"] = self.data_version
        
        if self.code_version:
            config["code_version"] = self.code_version
        
        if self.federated_rounds is not None:
            config["federated_rounds"] = self.federated_rounds
            config["clients_per_round"] = self.clients_per_round
            config["local_epochs"] = self.local_epochs
            config["aggregation_method"] = self.aggregation_method
        
        return config
    
    def get_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"{self.experiment_type}_{self.model}_seed{self.model_seed}"
    
    def get_group_name(self) -> str:
        return f"{self.experiment_id}_{self.experiment_type}_seed{self.model_seed}"
