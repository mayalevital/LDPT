#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:49:10 2021

@author: maya.fi@bm.technion.ac.il
"""
import torch
import matplotlib.pyplot as plt
#from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
from skimage.metrics import structural_similarity as ssim
#from pydicom import dcmread
import numpy as np
import SimpleITK as sitk

import numpy as np
from ipywidgets import interact, fixed

import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import math

import os
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode
from skimage.metrics import structural_similarity as ssim

import albumentations as A
import gdcm
from pydicom import dcmread
import numpy as np
import pickle
import torch.nn as nn
from timm.models.layers import trunc_normal_


def ModelParamsInit(model):
    assert isinstance(model, nn.Module)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def save_list(l, name):
    file_name = name 
    open_file = open(file_name, "wb")
    pickle.dump(l, open_file)
    open_file.close()
    
def load_list(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list
    #print(loaded_list)
    
def save_run(PATH, net, train_loss, valid_loss, valid_in_ssim, valid_res_ssim):
    torch.save(net, os.path.join(PATH, "net.pt"))
    save_list(train_loss, os.path.join(PATH, 'train_loss.pkl'))
    save_list(valid_loss, os.path.join(PATH, 'valid_loss.pkl'))
    save_list(valid_in_ssim, os.path.join(PATH, 'valid_in_ssim.pkl'))
    save_list(valid_res_ssim, os.path.join(PATH, 'valid_res_ssim.pkl'))
    
def test_ssim(LD, FD):
    s = ssim(stand_data(get_mat(LD)).numpy(), stand_data(get_mat(FD)).numpy())
    #print(s)
    if s>0.5 and s<1:
        return 1
    
def arrange_data(params, root_dir):
    dose_list = params['dose_list']
    chop = params['chop']

    df = pd.DataFrame(columns=['sub_ID', 'slice', 'Dose', 'LDPT', 'HDPT'])
    i=0
    for direct in root_dir:
        sub = os.listdir(direct)
        sub = [s for s in sub if s[-1] != 'p']       #remove zipped  
        for sub_dir in sub:
            sub_path = os.path.join(direct, sub_dir)
            sub_sub_path = os.listdir(sub_path)
            sub_sub_path = [s for s in sub_sub_path if s[-1] != 'X']       #remove zipped  
            for scans in sub_sub_path:
                sub_ID = scans[-6:] #pt. ID
                s_sub_path = os.path.join(direct, sub_dir, scans)
                d_scans = os.listdir(s_sub_path)
                FD_scan = [d for d in d_scans if (d[-6:] == 'NORMAL' or d[-6:] == 'normal' or d == 'Full_dose' or d == 'FD')]
                for FD in FD_scan:
                    FD_path = os.path.join(s_sub_path, FD)
                    for Dose in dose_list:
                        LD_scan = [d for d in d_scans if Dose in d][0]
                        LD_path = os.path.join(s_sub_path, LD_scan)
                        slices = os.listdir(LD_path)
                        #print(len(slices))
                        for sl in slices:
                            LD = os.path.join(LD_path, sl)
                            FD = os.path.join(FD_path, sl)
                            if(test_ssim(LD, FD)==1):
                                data = {'sub_ID': sub_ID, 'slice': sl, 'Dose': Dose, 'LDPT':LD, 'HDPT':FD}
                                df.loc[i] = data
                                i=i+1
                            else:
                                print('fail')
                            
                            #print(i)
                        #slices = []
                        
    return df
                    
                        
def split(lengths, data):
    sublists = []
    i = 0
    prev_index = 0
    for l in lengths:
        l=int(l)
        if(i==0):
            sublists.append(list(range(0, l-1)))
            
            i=i+1
            prev_index = l-1
        else:
            #curr_index = l + prev_index
            j=prev_index+1
            while(data.iloc[prev_index].sub_ID==data.iloc[j].sub_ID):
                j=j+1
                #print(data.iloc[prev_index].sub_ID)
            sublists.append(list(range(j, j+l-1)))
   
    return sublists                 

def get_mat(name):
    X = dcmread(name)
    X = X.pixel_array
    return torch.tensor(X.astype(float))            
    
def train_val_test_por(params, data):
    length = len(data)
    #my_list = list(range(1, length))
    weight_list = params['train_val_test'] 
    lengths = [np.ceil(i*length) for i in weight_list]
    
    sublists = split(lengths, data)
    #print()
    return sublists[0], sublists[1], sublists[2]

def scale_data(data):
    
    if(torch.std(data)==0):
        std=1
    else:
        std=torch.std(data)
    
    if(torch.max(data)!=0):
     
        data = (data-torch.min(data))/(torch.max(data)-torch.min(data))-torch.tensor(0.5)
   
    return data

def stand_data(data):
    
    if(torch.std(data)==0):
        std=1
    else:
        std=torch.std(data)
    data = (data-torch.mean(data))/(std)
 
    return data

def norm_data(data):
    norm=torch.norm(data, p='fro', dim=None, keepdim=False, out=None, dtype=None)
    if(norm==0):
        norm=1
    data = data/norm
    #print(norm)
    return data, norm

def calc_ssim(LDPT, NDPT):
    ssim_c = []
 
    s=LDPT.shape
    for i in range(s[0]):
        LD=LDPT[i][0]
        ND=NDPT[i][0]
        LDPT1 = stand_data(LD).numpy()
        NDPT1 = stand_data(ND).numpy()
                
        ssim_=ssim(NDPT1, LDPT1)
        ssim_c.append(ssim_)
        
    return sum(ssim_c)/len(ssim_c)
    

def plot_result(results, inputs, outputs, ssim_LD_ND, ssim_RE_ND):
    #scalebar = ScaleBar(0.08, "log L2 norm", length_fraction=0.25)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    
    ax1.imshow(inputs)
    ax1.set_title('LDPT')
    
    ax2.imshow(outputs)
    ax2.set_title('NDPT')
    
    ax3.imshow(results)
    ax3.set_title('NET')

    plt.show()
    
def load_model(N, PATH, trainloader_2):
        net = torch.load(os.path.join(PATH, 'net.pt'))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        std = []
        for i, data in enumerate(trainloader_2, 0):
          
            inputs = data['LDPT'].to(device)
            results = net(inputs)
           
            inputs = inputs[0,0,:,:].detach().cpu()
            #print(inputs.shape)

            outputs = data['NDPT'].to(device)
            outputs = outputs[0,0,:,:].detach().cpu() 
            results = results[0,0,:,:].detach().cpu() 
           

            #print("LDPT/NDPT", ssim(inputs.numpy(), outputs.numpy()))
            #print("results/NDPT", ssim(results.numpy(), outputs.numpy()))
            #out = outputs
            #mask = out
            #mask[mask<0.0000001] =1
            if(i%100==1):
                #plot_result(np.log(results-outputs)-np.log(outputs), np.log(inputs-outputs)-np.log(outputs), np.log(results-inputs)-np.log(outputs))
                plot_result(results, inputs, outputs, ssim(inputs.numpy(), outputs.numpy()), ssim(results.numpy(), outputs.numpy()))
        fig, (ax1, ax2) = plt.subplots(2)
        
        train_loss = load_list(os.path.join(PATH, 'train_loss.pkl'))
        valid_loss = load_list(os.path.join(PATH, 'valid_loss.pkl'))
        valid_in_ssim = load_list(os.path.join(PATH, 'valid_in_ssim.pkl'))
        valid_res_ssim = load_list(os.path.join(PATH, 'valid_res_ssim.pkl'))
    
        ax1.plot(range(0,N), train_loss, label = "training loss")
        ax1.plot(range(0,N), valid_loss, label = "validation loss")
        ax1.set(ylabel="loss")
        ax1.legend()
        ax1.set_title("training / validation loss over epochs")
        #axs[0].savefig(PATH + "train_valid_loss" + ".jpg")
        
        ax2.plot(range(0,N), valid_in_ssim, label = "LDPT/NDPT ssim")
        ax2.plot(range(0,N), valid_res_ssim, label = "result/NDPT ssim")
        ax2.set(ylabel="ssim")
        ax2.legend()
        ax2.set_title("validation ssim over epochs")
        fig.savefig(os.path.join(PATH, "_valid__loss_ssim.jpg"))
        #print(median(std))