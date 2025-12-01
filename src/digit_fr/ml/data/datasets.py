import torch
from torch.utils.data import DataLoader

class MultiTaskDataset: 
    def __init__(self, X, y_classification, y_probabilities=None, y_categories=None):
        self.X = torch.FloatTensor(X.values)
        self.y_classification = torch.FloatTensor(y_classification.values)
        self.y_probabilities = torch.FloatTensor(y_probabilities.values) if y_probabilities is not None else None
        self.y_categories = torch.LongTensor(y_categories.values) if y_categories is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        labels = {
            'classification': self.y_classification[idx]
        }

        if self.y_probabilities is not None:
            labels['probabilities'] = self.y_probabilities[idx]
        
        if self.y_categories is not None:
            labels['categories'] = self.y_categories[idx]
            
        return self.X[idx], labels

def create_data_loaders(data, batch_size=32, shuffle_train=True):
    train_dataset = MultiTaskDataset(data['train']['X'], data['train']['y_classification'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    
    loaders = [train_loader]
    
    if 'val' in data:
        val_dataset = MultiTaskDataset(data['val']['X'], data['val']['y_classification'])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        loaders.append(val_loader)
    
    if 'test' in data:
        y_test_probs = data['test'].get('y_probabilities', None)
        y_test_categories = data['test'].get('y_categories', None)
        test_dataset = MultiTaskDataset(data['test']['X'], data['test']['y_classification'], y_probabilities=y_test_probs, y_categories=y_test_categories)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        loaders.append(test_loader)
    
    if len(loaders) == 3:
        return train_loader, loaders[1], loaders[2]
    elif len(loaders) == 2:
        return train_loader, loaders[1]
    else:
        return train_loader