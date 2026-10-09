import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

SEED = 42
DATA_DIR = "data"
device = ("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_tf = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    normalize,  
])

def get_split(dataset):
    """
    Creates train, validation and test splits from the dataset.
    """
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(dataset))
    n = len(dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_indices = indices[ : n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val : ]

    return train_indices, val_indices, test_indices

def get_model():
    """
    Get the ResNet18 model with the final layer modified for 10 classes.
    """
    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(
        model.fc.in_features, 
        10
    )
    return model.to(device)

model = get_model()

