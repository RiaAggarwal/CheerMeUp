import torch.nn as nn

class LSloss:
    def __init__(self):
        self.mse = nn.MSELoss()
        self.real = 1
        self.fake = 0
    
    @staticmethod
    def discriminator_loss(self, d_real, d_gen):
        return self.mse(d_real, self.real) + self.mse(d_gen, self.fake)
    
    @staticmethod
    def generator_loss(self, dg_out):
        return self.mse(dg_out, self.real)
        