#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:21:36 2022

@author: maya.fi@bm.technion.ac.il
"""

#!/bin/sh
#chmod +x ./LDPTmain.sh
import os

from utilities import save_images, dict_mean, train_test_net, train_val_test_por, norm_data, plot_result, calc_ssim, ModelParamsInit, ModelParamsInit_unetr, save_run, gradient_magnitude, calc_valid_loss
from SGLD_ import SGLD
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
from utilities import load_model, laplacian_filter
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
import dataset
from dataset import ULDPT
plt.ioff()
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
            
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            #print("yes")
            m.train()  
            
def trainloaders(params, data):
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
#method = params['method'][0]
N_finish = params['N_finish']

if(t==1):
    opt = params['optimizer'][0]
    for wd in params['weight_decay']:
        for network in params['net']:
            for l in params['lambda']:
                for learn in params['lr']:
                    for method in params['method']:
                        print(network)
                        print('learning rate = ', learn) 
                        print('grads lambda = ', l)
                        print('optimizer ', opt)
                        print('N finish = ', N_finish)
                        print('method ', method)
                        print('alpha = ', params['alpha'])
                        print('weight decay = ', wd)
                        if network == 'unet':
                            #features=(32, 32, 32, 64, 128, 32)
                            if method == 'standard':
                                net = BasicUNet(spatial_dims=2, out_channels=1, features=(32, 32, 32, 64, 128, 32), norm=("group", {"num_groups": 4}), act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), dropout=params['dropout']).to(device)
                                ModelParamsInit(net)
                            if method == 'SGLD':
                                net = BasicUNet(spatial_dims=2, out_channels=1, features=(32, 32, 32, 64, 128, 32), norm=None, act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01})).to(device)
                                ModelParamsInit(net)
                                #optimizer = SGLD(params=net.parameters(), lr=learn, momentum=params['momentum'], weight_decay=wd, noise_scale=params['alpha'])
                     
                            if method == 'pre_train':
                                net = torch.load('/tcmldrive/users/Maya/Experiments_final/standard/unet_25_epochs_0.0005_lr_[0.7, 0.3]grad_loss_lambda/epoch24/net.pt')
                        if network == 'unetr':
                            net = UNETR2D(in_channels=1, out_channels=1, img_size=(352, 352), feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.2).to(device)
                            ModelParamsInit_unetr(net)
                        
                        criterion = nn.L1Loss()
                        
                        if opt == 'RMS':
                            optimizer=torch.optim.RMSprop(net.parameters(), lr=learn, alpha=0.99, eps=1e-08, weight_decay=wd, momentum=0, centered=False)
                        if opt == 'ADAM':
                            optimizer=torch.optim.Adam(net.parameters(), lr=learn, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
                        
                        [trainloader_1, trainloader_2] = trainloaders(params, data)
                        #print(len(trainloader_1))
                        train_test_net(trainloader_1, trainloader_2, network, N_finish, N, params, params['alpha'], learn, method, optimizer, criterion, net, device, l, wd)

          

if(t==2): 
    PATH = "/tcmldrive/users/Maya/Experiments/standard/epoch3/unet_5_epochs_0.0005_lr_[0.9, 0.1]grad_loss_lambda/"
    [trainloader_1, trainloader_2] = trainloaders(data)
    print(PATH)
    data = pd.read_pickle("./data_50.pkl", compression='infer')
    load_model(4, PATH, trainloader_2)

if(t==0):
    [trainloader_1, trainloader_2] = trainloaders(params, data)
    for epoch in range(1):  # loop over the dataset multiple times
    
        for i, data in enumerate(trainloader_2, 0):
 
            inputs = laplacian_filter(data['LDPT'],5)
            outputs = laplacian_filter(data['NDPT'],5)
            print(inputs.size())
  
            if(random.randint(1,10)==9):
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

                ax1.imshow(inputs[0,0,:,:])
                

                ax2.imshow(outputs[0,0,:,:])
                f.savefig(os.path.join('/tcmldrive/users/Maya/figs/', "img.jpg"))
                plt.close(f)
    
if(t==3):
  
    data = pd.read_pickle("./data_50.pkl", compression='infer')
    params = scan_params()
    PATH = '/tcmldrive/users/Maya/Experiments_fin/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    index_list = params['index_list']
    [trainloader_1, trainloader_2] = trainloaders(params, data)
    df = pd.DataFrame(columns=['method', 'LDPT', 'NDPT', 'idx', 'mean', 'std', 'ssim_0', 'ssim_net'])
    k=0
    for i, data in enumerate(trainloader_2, 0):
         #if i in index_list:
        inputs = data['LDPT'].to(device)
        NDPT = data['NDPT']
        NDPT = NDPT[0,0,:,:].numpy()
       
        for direct in os.listdir(PATH):
            
            if direct == 'SGLD':
                print(direct)
                j=0
                df_SGLD = pd.DataFrame(columns=['method', 'epoch', 'idx', 'LDPT', 'NDPT', 'NET'])
                for epoch_test in os.listdir(os.path.join(PATH, direct)):
                    for epoch_num in os.listdir(os.path.join(PATH, direct, epoch_test)):
                        path = os.path.join(os.path.join(PATH, direct, epoch_test, epoch_num), 'net.pt')
                        if os.path.isfile(path):
                            net = torch.load(path)
                            NET = net(inputs)
                            LDPT = inputs[0,0,:,:].detach().cpu().numpy()
                            NET = NET[0,0,:,:].detach().cpu().numpy()
                            data = {'method': direct, 'epoch': epoch_num, 'idx': i, 'LDPT':LDPT, 'NDPT':NDPT, 'NET':NET}
                            df_SGLD.loc[j] = data
                            j=j+1
                    df_return = dict_mean(df_SGLD, i)
                    df.loc[k] = {'method': direct, 'LDPT':LDPT, 'NDPT':NDPT, 'idx':i, 'mean':df_return['mean'], 'std':df_return['std'], 'ssim_0':df_return['ssim_0'], 'ssim_net':df_return['ssim_net']}
                    k=k+1
                    out_dirs = os.path.join(PATH, direct, epoch_test, 'STD_maps_recon')
                    save_images(df_return, out_dirs)
   
            if direct == 'standard':
                print(direct)    
                for epoch_test in os.listdir(os.path.join(PATH, direct)):
                        epoch_num = os.listdir(os.path.join(PATH, direct, epoch_test))[-1]
                        df_standard = pd.DataFrame(columns=['method', 'epoch', 'idx', 'LDPT', 'NDPT', 'NET'])
                        path = os.path.join(os.path.join(PATH, direct, epoch_test, epoch_num), 'net.pt')
                        if os.path.isfile(path):
                            for j in range(0, N_finish):
                                net = torch.load(path)         
                                net.eval()
                                enable_dropout(net)
                                NET = net(inputs)
                                LDPT = inputs[0,0,:,:].detach().cpu().numpy()
                                NET = NET[0,0,:,:].detach().cpu().numpy()
                                data = {'method': direct, 'epoch': j, 'idx': i, 'LDPT':LDPT, 'NDPT':NDPT, 'NET':NET}
                                df_standard.loc[j] = data
                            
    
                        df_return = dict_mean(df_standard, i)
                        df.loc[k] = {'method': direct, 'LDPT':LDPT, 'NDPT':NDPT, 'idx':i, 'mean':df_return['mean'], 'std':df_return['std'], 'ssim_0':df_return['ssim_0'], 'ssim_net':df_return['ssim_net']}
                        k=k+1
                        out_dirs = os.path.join(PATH, direct, epoch_test, epoch_num, 'STD_maps_recon')
                        save_images(df_return, out_dirs)
         
                
    df.to_pickle(os.path.join(PATH, 'results.pkl'), compression='infer', protocol=5, storage_options=None)
    
    
if t==4:
    path = '/tcmldrive/users/Maya/Experiments_fin/results.pkl'
    df = pd.read_pickle(path)
    df_SGLD = df[df['method']=='SGLD']
    df_standard = df[df['method']=='standard']
    print('SGLD mean SSIM', df_SGLD.ssim_net.to_numpy().mean())
    print('Dropout mean SSIM', df_standard.ssim_net.to_numpy().mean())