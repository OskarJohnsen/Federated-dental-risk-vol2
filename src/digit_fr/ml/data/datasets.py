import torch
from torch.utils.data import DataLoader

class MultiTaskDataset: 
    def __init__(self, X, y_classification):
        self.X = torch.FloatTensor(X.values)
        self.y_classification = torch.FloatTensor(y_classification.values)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], {
            'classification': self.y_classification[idx]
        }

def create_data_loaders(data, batch_size=32, shuffle_train=True):
    train_dataset = MultiTaskDataset(data['train']['X'], data['train']['y_classification'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    
    if 'test' in data:
        test_dataset = MultiTaskDataset(data['test']['X'], data['test']['y_classification'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    
    return train_loader