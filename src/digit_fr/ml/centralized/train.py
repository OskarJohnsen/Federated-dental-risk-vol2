import torch
from torch.optim.lr_scheduler import StepLR
from ..models.base.trainer import BaseTrainer
from ..models.architectures.mlp import MLP
from ..data.loaders import load_data_with_split
from ..data.datasets import create_data_loaders

def main():
    print("CENTRALIZED TRAINING EXPERIMENT")

    print("\nData")
    data = load_data_with_split()
    n_features = data['train']['X'].shape[1]
    print(f"Features: {n_features}")
    print(f"Train samples: {len(data['train']['X'])}")
    print(f"Test samples: {len(data['test']['X'])}")
    
    # class weights for weighted CLE
    y_class_train = data['train']['y_class']
    n_class_0 = (y_class_train == 0).sum()
    n_class_1 = (y_class_train == 1).sum()
    weight_class_0 = n_class_1 / n_class_0
    
    train_loader, test_loader = create_data_loaders(data, batch_size=32)
    
    model = MLP(
        input_size=n_features,
        hidden_size=[128, 64],
        dropout=0.2,
        n_clf_classes=2,
        n_reg_targets=4
    )
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
     
    class_weights = torch.tensor([weight_class_0, 1.0], dtype=torch.float32)
    loss_clf = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    loss_reg = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        loss_clf=loss_clf,
        loss_reg=loss_reg,
        scheduler=scheduler,
        experiment_type='centralized',
        seed=42
    )
    
    print(f"Training on device: {trainer.device}")
    
    results = trainer.fit(train_loader, val_loader=test_loader, epochs=55)
    
    print("EVALUATION")
    final_metrics = trainer.evaluate(test_loader)
    
    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    print(f"\nCheckpoint saved: {results['checkpoint_path']}")
    print("\nDone")

if __name__ == '__main__':
    main()