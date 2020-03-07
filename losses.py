import torch.nn as nn
import torch

class LSloss:
    
    
    def __init__(self):
        pass
    
    @staticmethod
    def discriminator_loss(d_real, d_gen):
        mse = nn.MSELoss()
        return mse(d_real, torch.ones_like(d_real)) + mse(d_gen, torch.zeros_like(d_gen))
    
    @staticmethod
    def generator_loss(dg_out):
        mse = nn.MSELoss()
        return mse(dg_out, torch.ones_like(dg_out))
        