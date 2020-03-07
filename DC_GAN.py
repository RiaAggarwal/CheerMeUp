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

batch_size = args["batch_size"]
nz = 100
ngf = 64
nc = 3
ngpu = 1

ndf = 64

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


generator_ = generator.Generator(nz, ngf, nc, ngpu).to(device)
discriminator_ = discriminator.Discriminator(nc, ndf).to(device)

generator_.apply(weights_init)
discriminator_.apply(weights_init)

params_gen = list(generator_.parameters())
params_dis = list(discriminator_.parameters())

optimizer_gen = torch.optim.Adam(params_gen, lr=args['learning_rate_gen'])
optimizer_dis = torch.optim.Adam(params_dis, lr=args['learning_rate_dis'])

loss = nn.MSELoss()
ones = torch.ones(batch_size,1,1,1).to(device)
zeros = torch.zeros(batch_size,1,1,1).to(device)

for epoch in range(args['epochs']):
    generator_.train()
    discriminator_.train()
    
    for idx, (imgs) in enumerate(dataloader):
        discriminator_.zero_grad()
        generator_.zero_grad()
        x_real = imgs[:,:3, :, :].to(device)
        print(x_real.size())
        z = torch.randn(imgs.shape[0], nz, 1, 1, device=device) 
        
        x_gen = generator_(z)
        dg_out = discriminator_(x_gen)
        dr_out = discriminator_(x_real)
        print(dg_out.size())
        
        g_loss = loss(dg_out, ones)
        d_loss = loss(dr_out, ones)+ loss(dg_out, zeros)
        

        g_loss.backward()
        d_loss.backward()
        
        optimizer_gen.step()
        optimizer_dis.step()
        
        print(f"epoch :{epoch}, g_loss : {g_loss.mean().item()}, d_loss : {d_loss.mean().item()}")

        
        
        
        
    