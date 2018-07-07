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
parser.add_argument("--epoch", type = int,
                    required = True)

args = parser.parse_args()

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)
    
train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']

batch_size = 1
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data, batch_size = batch_size)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

if args.num_stochastic_layers == 1:
    vae = IWAE_1(50, 784)
elif args.num_stochastic_layers == 2:
    vae = IWAE_2(100, 50, 784)
    
vae.double()
vae.cuda()
model_file_name = ("./output/model/"
                   "{}_layers_{}_k_{}_epoch_{}.model").format(
                       args.model, args.num_stochastic_layers,
                       args.num_samples,args.epoch)
vae.load_state_dict(torch.load(model_file_name))

tot_loss = 0
tot_size = 0
for idx, data in enumerate(test_data_loader):
    print(idx)
    data = data.double()
    with torch.no_grad():
        inputs = Variable(data).cuda()
        inputs = inputs.expand(5000, batch_size, 784)    
        loss = vae.test_loss(inputs)

        size = inputs.size()[0]
        tot_size += size
        tot_loss += loss.item() * size

print(model_file_name, end = ",")        
print("Average loss: {:.2f}".format(tot_loss/tot_size))
