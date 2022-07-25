#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:21:36 2022

@author: maya.fi@bm.technion.ac.il
"""

#!/bin/sh
#chmod +x ./LDPTmain.sh
import os
import dataset
from dataset import ULDPT
from utilities import train_val_test_por, norm_data, plot_result, calc_ssim, ModelParamsInit, ModelParamsInit_unetr, save_run, gradient_magnitude, calc_valid_loss
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
from utilities import arrange_data, arrange_data_siemense
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
import random
from torchsummary import summary
import time
import monai
from monai.networks.blocks import UnetBasicBlock
from monai.networks.nets import UNETR, UNet
from monai.optimizers import LearningRateFinder
import unet_2
from unet_2 import BasicUNet
from unetr import UNETR2D
import pytorch_ssim
plt.ioff()

def trainloaders(data):
    _dataset = ULDPT(data)
    
    train_por, val_por = train_val_test_por(params, data)
    print("train portion size = ", len(train_por))
    print("test portion size = ", len(val_por))
    
    train_set = torch.utils.data.Subset(_dataset, train_por)
    val_set = torch.utils.data.Subset(_dataset, val_por)
       
    trainloader_1 = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
                                                shuffle=True, num_workers=4)
    trainloader_2 = torch.utils.data.DataLoader(val_set, batch_size=params['batch_size'],
                                                shuffle=True, num_workers=4)

    return trainloader_1, trainloader_2


CUDA_VISIBLE_DEVICES=1 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
params = scan_params()

data = pd.read_pickle("./data_50.pkl", compression='infer')


t=params['t']
N = params['num_of_epochs']
l = params['lambda']
print("small weight decay, un-filtered data")
if(t==1):
    for opt in params['optimizer']:
        for network in params['net']:
            for l in params['lambda']:
                for learn in params['lr']:
                    print(network)
                    print('learning rate=', learn) 
                    print('grads lambda=', l)
                    print('optimizer ', opt)
                    PATH = network + '_' + str(params['num_of_epochs']) + "_epochs_" + str(learn) + "_lr_" + str(l) + "grad_loss_lambda"
                    if not os.path.exists(PATH):
                        os.makedirs(PATH)
                    if network == 'unet':
                        net = BasicUNet(spatial_dims=2, out_channels=1, features=(32, 32, 32, 64, 128, 32), norm=("group", {"num_groups": 4}), act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), dropout=0.1).to(device)
                        ModelParamsInit(net)
                    if network == 'unetr':
                        net = UNETR2D(in_channels=1, out_channels=1, img_size=(352, 352), feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.2).to(device)
                        ModelParamsInit_unetr(net)

                    criterion = nn.L1Loss()
                    if opt == 'RMS':
                        optimizer=torch.optim.RMSprop(net.parameters(), lr=learn, alpha=0.99, eps=1e-08, weight_decay=params['weight_decay'], momentum=0, centered=False)
                    if opt == 'ADAM':
                        optimizer=torch.optim.Adam(net.parameters(), lr=learn, betas=(0.9, 0.999), eps=1e-08, weight_decay=params['weight_decay'])

                    valid_in_ssim = []
                    valid_res_ssim = []
                    [trainloader_1, trainloader_2] = trainloaders(data)

                    valid_loss = []
                    #valid_loss.append(calc_valid_loss(criterion, l, data, device, trainloader_2))
                    train_loss = []
                    for epoch in range(N):  # loop over the dataset multiple times
                        print('epoch ', epoch+1, ' out of ', N)
                        
                        running_train_loss = 0.0
                        running_g_loss = 0.0
                        running_l1_loss = 0.0
                        SSIM_LDPT_NDPT_train = []
                        SSIM_RESU_NDPT_train = []
                        SSIM_LDPT_NDPT_valid = []
                        SSIM_RESU_NDPT_valid = []
                        
                        net.train()
                        for i, data_train in enumerate(trainloader_1, 0):
                            
                            inputs = data_train['LDPT'].to(device)
                            outputs = data_train['NDPT'].to(device)
                            
                            optimizer.zero_grad()
                            results = net(inputs)
                            #print(results)
                            l1_train = torch.tensor(l[0])*criterion(results, outputs)
                            grad_train = torch.tensor(l[1])*criterion(gradient_magnitude(results), gradient_magnitude(outputs))
                            ssim_value = 1 - calc_ssim(results.detach().cpu(), outputs.detach().cpu())
                            if ssim_value > 0.05:
                                #print("ssim_value")
                                loss_train = l1_train + grad_train
                                loss_train.backward()
                                optimizer.step()
                                #pull out the gradient
                                #add the noise gaussian with alpha std
                                running_train_loss = running_train_loss + loss_train.item()
                                running_g_loss = running_g_loss + grad_train.item()
                                running_l1_loss = running_l1_loss + l1_train.item()
                                SSIM_LDPT_NDPT_train.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
                                SSIM_RESU_NDPT_train.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
 
                        running_valid_loss = 0.0
                        with torch.no_grad():
                            
                            net.eval()
                            for i, data_val in enumerate(trainloader_2, 0):
                                inputs = data_val['LDPT'].to(device)   
                                outputs = data_val['NDPT'].to(device)
                                #print(inputs)
                                #print(inputs.float())
                                results = net(inputs)
                                l1_val = torch.tensor(l[0])*criterion(outputs, results)
                                grad_val = torch.tensor(l[1])*criterion(gradient_magnitude(outputs), gradient_magnitude(results))
                                
                           
                                loss_val = l1_val + grad_val      
                                running_valid_loss = running_valid_loss + loss_val.item()
                                SSIM_LDPT_NDPT_valid.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
                                SSIM_RESU_NDPT_valid.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
                         
                            print("ssim valid LDPT/NDPT", np.mean(SSIM_LDPT_NDPT_valid))
                            valid_in_ssim.append(np.mean(SSIM_LDPT_NDPT_valid))
                            print("ssim valid results/NDPT", np.mean(SSIM_RESU_NDPT_valid))
                            valid_res_ssim.append(np.mean(SSIM_RESU_NDPT_valid))
                            print('[%d, %5d] training loss: %.5f' %
                                          (epoch + 1, i + 1, running_train_loss))
                            print('[%d, %5d] training grad loss: %.5f' %
                                          (epoch + 1, i + 1, running_g_loss))
                            print('[%d, %5d] training l1 loss: %.5f' %
                                          (epoch + 1, i + 1, running_l1_loss))
                            train_loss.append(running_train_loss)
                            print('[%d, %5d] validation loss: %.5f' %
                                          (epoch + 1, i + 1, running_valid_loss))
                            valid_loss.append(running_valid_loss)
                
                
                print('Finished Training')
                 
                save_run(PATH, net, train_loss, valid_loss, valid_in_ssim, valid_res_ssim)
                load_model(N, PATH, trainloader_2)
if(t==2): 
    PATH = 'unet_40_epochs_0.0005_lr_[0.9, 0.1]grad_loss_lambda'
    print(PATH)
    load_model(N, PATH, trainloader_2)

if(t==0):
  
    for epoch in range(N):  # loop over the dataset multiple times
    
        for i, data in enumerate(trainloader_2, 0):
 
            inputs = data['LDPT'].to(device)
            outputs = data['NDPT'].to(device)
            print(inputs.size())
  
            if(random.randint(1,10)==9):
                plt.figure()
                plt.subplot(1,2,1)
                #print(data['NDPT'].size()[0])
                #m = torch.mean(data['NDPT'][0,0,:,:,0])
                #print(m)
                #plt.title(str(m))
                plt.imshow(data['LDPT'][0,0,:,:])
                plt.subplot(1,2,2)
                #m = torch.mean(data['LDPT'][0,0,:,:,0])
                #print(m)
                #plt.title(str(m))
                plt.imshow(data['NDPT'][0,0,:,:])
                m=[]
                m1=[]
                #for idx in range(data['NDPT'].size()[0]):  	
                    #m.append(torch.mean(data['NDPT'][idx,0,:,:,0]))
                    #m1.append(torch.mean(data['LDPT'][idx,0,:,:,0]))
                
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