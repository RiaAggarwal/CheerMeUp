import torch 
import torchvision
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
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = get_loader('./file_names.csv', transform, batch_size=args['batch_size'], shuffle=True, 
                        num_workers=args['num_workers'])

d_cp = 'model_cps/generator_epoch_latest.pth'
g_cp = 'model_cps/generator_epoch_latest.pth'

batch_size = args["batch_size"]


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

try :
    generator_.load_state_dict(torch.load(g_cp))
    discriminator_.load_state_dict(torch.load(d_cp))
except:
    pass


params_gen = list(generator_.parameters())
params_dis = list(discriminator_.parameters())

optimizer_gen = torch.optim.Adam(params_gen, lr=args['learning_rate_gen'], betas=(args['beta1'], 0.999))
optimizer_dis = torch.optim.Adam(params_dis, lr=args['learning_rate_dis'], betas=(args['beta1'], 0.999))

loss = nn.MSELoss()
ones = torch.ones(batch_size,1,1,1).to(device)
zeros = torch.zeros(batch_size,1,1,1).to(device)

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device) 


for epoch in range(args['epochs']):
    generator_.train()
    discriminator_.train()
    g_loss_arr = []
    d_loss_arr = []
    
    t1 = time.time()
    
    for idx, (imgs) in enumerate(dataloader):
        discriminator_.zero_grad()
        generator_.zero_grad()
        x_real = imgs[:,:3, :, :].to(device)
        #print(x_real.size())
        z = torch.randn(batch_size, nz, 1, 1, device=device) 
        
        dr_out = discriminator_(x_real)
        d_loss_real = loss(dr_out, ones)
        d_loss_real.backward()

        x_gen = generator_(z)
        dg_out = discriminator_(x_gen.detach())
        d_loss_fake = loss(dg_out, zeros)
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        optimizer_dis.step()

        
        #print(dg_out.size())
        
        # g_loss = loss(dg_out, ones)
        # d_loss = loss(dr_out, ones) + loss(dg_out, zeros)
        
        dg_out = discriminator_(x_gen)
        g_loss = loss(dg_out, ones)
        g_loss.backward()
        optimizer_gen.step()
        g_loss_arr.append(g_loss.item())
        d_loss_arr.append(d_loss.item())
        
    t2 = time.time()
        
    print(f"epoch :{epoch}, g_loss : {np.array(g_loss_arr).mean()}, d_loss : {np.array(d_loss_arr).mean()}, took {t2-t1} seconds")

    # do checkpointing
    torch.save(generator_.state_dict(), f'model_cps/generator_epoch_{epoch}.pth')
    torch.save(discriminator_.state_dict(), f'model_cps/discriminator_epoch_{epoch}.pth')
    
    torch.save(generator_.state_dict(), 'model_cps/generator_epoch_latest.pth')
    torch.save(discriminator_.state_dict(), 'model_cps/discriminator_epoch_latest.pth')
    
    if (epoch %3) == 0 :
        fake = generator_(fixed_noise)
        torchvision.utils.save_image(fake.detach(),f'results/generated/fake_samples_epoch_{epoch}.png',normalize=True)
        
        
        
        
        
    