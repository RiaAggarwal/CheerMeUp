import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import os 
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from dataloader_GAN import get_loader
import matplotlib.pyplot as plt
from config import *
from losses import LSloss 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = get_loader('file_names.csv', transform, batch_size=args['batch_size'], shuffle=True, 
                        num_workers=args['num_workers'])

generator = .....
discriminator = ....

params_gen = list(generator.params())
params_dis = list(discriminator.params())

optimizer_gen = torch.optim.Adam(params_gen, lr=args['learning_rate_gen'])
optimizer_dis = torch.optim.Adam(params_dis, lr=args['learning_rate_dis'])

for epoch in range(args['epochs']):
    generator.train()
    discriminator.train()
    
    for idx, (imgs) in enumerate(dataloader):
        x_real = imgs.to(device)
        z = torch.randn(imgs.shape[0],args['noise_len'])
        
        x_gen = generator(z)
        dg_out = discriminator(x_gen)
        dr_out = discriminator(x_real)
        
        g_loss = LSloss.generator_loss(dg_out)
        d_loss = LSloss.discriminator_loss(dr_out,dg_out)
        
        discriminator.zero_grad()
        generator.zero_grad()
        
        g_loss.backward()
        d_loss.backward()
        
        optimizer_gen.step()
        optimizer_dis.step()
        
        
        
        
    