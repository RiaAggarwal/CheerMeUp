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
from models import generator,discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = get_loader('./file_names.csv', transform, batch_size=args['batch_size'], shuffle=True, 
                        num_workers=args['num_workers'])


nz = 100
ngf = 64
nc = 3
ngpu = 1

input_size= 64
ndf = 64


generator_ = generator.Generator(nz, ngf, nc, ngpu).to(device)
discriminator_ = discriminator.Discriminator(input_size, nc, ndf).to(device)

params_gen = list(generator_.parameters())
params_dis = list(discriminator_.parameters())

optimizer_gen = torch.optim.Adam(params_gen, lr=args['learning_rate_gen'])
optimizer_dis = torch.optim.Adam(params_dis, lr=args['learning_rate_dis'])

for epoch in range(args['epochs']):
    generator_.train()
    discriminator_.train()
    
    for idx, (imgs) in enumerate(dataloader):
        discriminator_.zero_grad()
        generator_.zero_grad()
        
        x_real = imgs.to(device)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        z = torch.randn(imgs.shape[0],nz).to(device)
        
        x_gen = generator_(z)
        dg_out = discriminator_(x_gen)
        dr_out = discriminator_(x_real)
        
        g_loss = LSloss.generator_loss(dg_out)
        d_loss = LSloss.discriminator_loss(dr_out,dg_out)
        

        
        g_loss.backward()
        d_loss.backward()
        
        optimizer_gen.step()
        optimizer_dis.step()
        
        print(f"epoch :{epoch}, g_loss : {g_loss.item()}, d_loss : {d_loss.item()}")

        
        
        
        
    