import random
import numpy as np
import torch

def all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    return seed

def data_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return seed