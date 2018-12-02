__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/10/16 02:50:08"

import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
sys.path.append("../model/")
from vae_models import *
from sys import exit
import argparse

parser = argparse.ArgumentParser(description="Importance Weighted Auto-Encoder")
parser.add_argument("--model", type = str,
                    choices = ["IWAE", "VAE"],
                    required = True,
                    help = "choose VAE or IWAE to use")
parser.add_argument("--num_stochastic_layers", type = int,
                    choices = [1, 2],
                    required = True,
                    help = "num of stochastic layers used in the model")
parser.add_argument("--num_samples", type = int,
                    required = True,
                    help = """num of samples used in Monte Carlo estimate of 
                              ELBO when using VAE; num of samples used in 
                              importance weighted ELBO when using IWAE.""")
args = parser.parse_args()

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']

batch_size = 1000
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data, batch_size = batch_size,
                               shuffle = True)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

if args.num_stochastic_layers == 1:
    vae = IWAE_1(50, 784)
elif args.num_stochastic_layers == 2:
    vae = IWAE_2(100, 50, 784)
    
vae.double()
vae.cuda()

optimizer = optim.Adam(vae.parameters())
num_epoches = 5000
train_loss_epoch = []
for epoch in range(num_epoches):
    running_loss = []    
    for idx, data in enumerate(train_data_loader):
        data = data.double()
        inputs = Variable(data).cuda()
        if args.model == "IWAE":
            inputs = inputs.expand(args.num_samples, batch_size, 784)
        elif args.model == "VAE":
            inputs = inputs.repeat(args.num_samples, 1)
            inputs = inputs.expand(1, batch_size*args.num_samples, 784)
            
        optimizer.zero_grad()
        loss = vae.train_loss(inputs)
        loss.backward()
        optimizer.step()    
        print(("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}")
              .format(epoch, idx, loss.item()), flush = True)
        running_loss.append(loss.item())


    train_loss_epoch.append(np.mean(running_loss))

    if (epoch + 1) % 1000 == 0:
        torch.save(vae.state_dict(),
                   ("./output/model/{}_layers_{}_k_{}_epoch_{}.model")
                   .format(args.model, args.num_stochastic_layers,
                           args.num_samples, epoch))

