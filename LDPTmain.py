#!/bin/sh
#chmod +x ./LDPTmain.sh
import os
import dataset
from dataset import ULDPT
import u_net_torch
from u_net_torch import Net
from utilities import train_val_test_por
from utilities import norm_data
from utilities import plot_result
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from params import scan_params
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from utilities import load_model
from utilities import arrange_data
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
import random
from torchsummary import summary
import time
import UNTER.UnetTr
from UNTER.UnetTr import UNETR
import monai
from monai.networks.blocks import UnetBasicBlock
#from monai.networks.nets import BasicUNet
from monai.optimizers import LearningRateFinder
import unet_2
from unet_2 import BasicUNet
from utilities import calc_ssim
from utilities import ModelParamsInit
plt.ion()


CUDA_VISIBLE_DEVICES=1 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


params = scan_params()
num_chan = params['num_chan']
root_dir = ['/tcmldrive/databases/Public/uExplorer/']

#data = arrange_data(params, root_dir)
#print(data.head)
#data.to_pickle("./data_100.pkl", compression='infer', protocol=4)
data = pd.read_pickle("./data_50.pkl", compression='infer')
_dataset = ULDPT(data, root_dir, params)


train_por, val_por, test_por = train_val_test_por(params, data)
#print(train_por)
train_set = torch.utils.data.Subset(_dataset, train_por)
val_set = torch.utils.data.Subset(_dataset, val_por)
test_set = torch.utils.data.Subset(_dataset, test_por)

trainloader_1 = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
                                            shuffle=True, num_workers=8)
trainloader_2 = torch.utils.data.DataLoader(val_set, batch_size=params['batch_size'],
                                            shuffle=True, num_workers=8)
trainloader_3 = torch.utils.data.DataLoader(test_set, params['batch_size'], shuffle=True, num_workers=8)

l = []
t=1
N = params['num_of_epochs']
PATH = str(params['num_of_epochs']) + "_epochs_" + str(params['num_kernels']) + "_kernels_" + str(params['num_chan']) + "_chan_" + str(params['multi_slice_n']) + "_slices" + ".pt"
if(t==1):
    
    net = BasicUNet(spatial_dims=2, out_channels=1, features=(16, 16, 16, 32, 64, 16), norm=("group", {"num_groups": 4}), act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), dropout=0.05)
   
    net.to(device)
    print(summary(net, input_size=(1, 360, 360)))
    ModelParamsInit(net)
    criterion = nn.L1Loss()
    optimizer=torch.optim.RMSprop(net.parameters(), lr=params['lr'], alpha=0.99, eps=1e-08, weight_decay=0.005, momentum=0, centered=False)

    valid_in_ssim = []
    valid_res_ssim = []
    train_loss = []
    valid_loss = []
    for epoch in range(N):  # loop over the dataset multiple times
        print(N)
        running_train_loss = 0.0
        SSIM_LDPT_NDPT_train = []
        SSIM_RESU_NDPT_train = []
        SSIM_LDPT_NDPT_valid = []
        SSIM_RESU_NDPT_valid = []
        for i, data in enumerate(trainloader_1, 0):
 
            inputs = data['LDPT'].to(device)
            outputs = data['NDPT'].to(device)
            
            optimizer.zero_grad()
            results = net(inputs)
               
            loss = criterion(results, outputs)
            loss.backward()
            optimizer.step()
            running_train_loss = running_train_loss + loss.item()
            SSIM_LDPT_NDPT_train.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
            SSIM_RESU_NDPT_train.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
        net.train()
             
        running_valid_loss = 0.0
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(trainloader_2, 0):
                inputs = data['LDPT'].to(device)   
                outputs = data['NDPT'].to(device)
                #print(inputs)
                #print(inputs.float())
                results = net(inputs)
                loss = criterion(results, outputs)        
                running_valid_loss = running_valid_loss + loss.item()
                SSIM_LDPT_NDPT_valid.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
                SSIM_RESU_NDPT_valid.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
     
        print("ssim valid LDPT/NDPT", np.mean(SSIM_LDPT_NDPT_valid))
        valid_in_ssim.append(np.mean(SSIM_LDPT_NDPT_valid))
        print("ssim valid results/NDPT", np.mean(SSIM_RESU_NDPT_valid))
        valid_res_ssim.append(np.mean(SSIM_RESU_NDPT_valid))
        print('[%d, %5d] training loss: %.3f' %
                      (epoch + 1, i + 1, running_train_loss))
        train_loss.append(running_train_loss)
        print('[%d, %5d] validation loss: %.3f' %
                      (epoch + 1, i + 1, running_valid_loss))
        valid_loss.append(running_valid_loss)
    print('Finished Training')
 
    torch.save(net, PATH)
 
if(t==0): 
    load_model(PATH, trainloader_2)

if(t==2):
  
    for epoch in range(N):  # loop over the dataset multiple times
    
        for i, data in enumerate(trainloader_2, 0):
 
            inputs = data['LDPT'].to(device)
            outputs = data['NDPT'].to(device)
            print(inputs.size())
  
            if(random.randint(1,1000)==9):
                plt.figure()
                plt.subplot(1,2,1)
                print(data['NDPT'].size()[0])
                #m = torch.mean(data['NDPT'][0,0,:,:,0])
                #print(m)
                #plt.title(str(m))
                plt.imshow(data['LDPT'][0,0,:,:,0])
                plt.subplot(1,2,2)
                #m = torch.mean(data['LDPT'][0,0,:,:,0])
                #print(m)
                #plt.title(str(m))
                plt.imshow(data['NDPT'][0,0,:,:,0])
                m=[]
                m1=[]
                for idx in range(data['NDPT'].size()[0]):  	
                    m.append(torch.mean(data['NDPT'][idx,0,:,:,0]))
                    m1.append(torch.mean(data['LDPT'][idx,0,:,:,0]))
                print("NDPT")
                print(m)
                print("LDPT")
                print(m1)
if(t==3):
  
    for epoch in range(N):  # loop over the dataset multiple times
    
        for i, data in enumerate(trainloader_1, 0):
 
            m = []
            m1 = []
            for idx in range(data['NDPT'].size()[0]):  	
                m.append(torch.mean(data['NDPT'][idx,0,:,:,0]))
                m1.append(torch.mean(data['LDPT'][idx,0,:,:,0]))
            print("NDPT")
            print(m)
            print("LDPT")
            print(m1)
