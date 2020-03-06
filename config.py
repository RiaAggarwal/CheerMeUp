import torch.nn as nn
import pickle

baseDataPath = './data/'

from torchvision import transforms

size = (64,64)

transform = transforms.Compose([transforms.Resize(size), 
                                transforms.ToTensor()])

args = {
    'epochs': 100,
    'batch_size': 4,
    'num_workers': 16,
    'learning_rate_gen': 1e-3,
    'learning_rate_dis': 1e-3,
    'noise_len': 100
}

