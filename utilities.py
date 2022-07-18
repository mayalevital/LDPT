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
#from ipywidgets import interact, fixed

import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import math

import os
import torch
import torch.nn
#import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import warnings
warnings.filterwarnings("ignore")
plt.ioff()   # interactive mode
from skimage.metrics import structural_similarity as ssim

import albumentations as A
import gdcm
from pydicom import dcmread
import numpy as np
import pickle
import torch.nn as nn
#import torch.functional.nn as F
from timm.models.layers import trunc_normal_

import kornia
import torch
from torch import Tensor
import monai
from monai.networks.blocks import Convolution
import unet_2
from unet_2 import fin_conv
import albumentations as A
import h5py


def transforma():
    transform = A.Compose([A.Flip(p=0.5)], additional_targets={'image0': 'image', 'image1': 'image'})
    return transform

def export_pixel_array(in_file_LD, out_file_LD, dataset_LD, in_file_FD, out_file_FD, dataset_FD):
    LDPT = dcmread(in_file_LD).pixel_array.astype(float)
    NDPT = dcmread(in_file_FD).pixel_array.astype(float)
    transforms = transforma()
    transformed = transforms(image=LDPT, image0=NDPT)
    LDPT = transformed["image"]
    NDPT = transformed["image0"]
    #print(norm_L.size())
    LDPT = norm_data(LDPT[4:356, 4:356])
    NDPT = norm_data(NDPT[4:356, 4:356])
    
    h5 = h5py.File(out_file_LD)
    h5.create_dataset(dataset_LD, data=LDPT)
    h5.close()
    
    h5 = h5py.File(out_file_FD)
    h5.create_dataset(dataset_FD, data=NDPT)
    h5.close()
    

def calc_valid_loss(criterion, l, data, device, trainloader_2):
    running_valid_loss = 0.0
    with torch.no_grad():
     
        for i, data in enumerate(trainloader_2, 0):
            
            inputs = data['LDPT'].to(device)
            outputs = data['NDPT'].to(device)
            loss_ = torch.tensor(l[0])*criterion(outputs, inputs)
            loss_grad = torch.tensor(l[1])*criterion(gradient_magnitude(outputs), gradient_magnitude(inputs))
           
           
            loss_v = loss_ + loss_grad    
            #print(loss_v.item())
            running_valid_loss = running_valid_loss + loss_v.item()
        return running_valid_loss

def gradient_magnitude(x):
    x_grad = kornia.filters.spatial_gradient(x, mode='diff', order=1)
    x_grad = torch.squeeze(x_grad)
    x_grad_mag = torch.sqrt(torch.square(x_grad[0]) + torch.square(x_grad[1]))
    return x_grad_mag
    
def ModelParamsInitHelper(m, flag):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and flag == "LeakyRELU":
        #print("init weight LeakyRELU")
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif isinstance(m, nn.Conv2d) and flag == "Tanh":
        #print("update weight conv Tanh")
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
    elif isinstance(m , (nn.GroupNorm, nn.LayerNorm)):
        #print("init weight norm")
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
   
    #else:
        #print("not")


def ModelParamsInit(model):
    assert isinstance(model, nn.Module)
    for name, m in model.named_modules():
       for name, mm in m.named_modules():
            for name, mmm in mm.named_modules():
                for name, mmmm in mmm.named_modules():
                    if isinstance(mmmm, (monai.networks.blocks.Convolution)):
                        for mmmmm in mmmm:
                            if isinstance(mmmmm, nn.Conv2d):
                                torch.nn.init.kaiming_uniform_(mmmmm.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                            else:
                                for mmmmmm in mmmmm:
                                    ModelParamsInitHelper(mmmmmm, "LeakyRELU")
                    elif isinstance(mmmm, (unet_2.fin_conv)):
                        for mmmmm in mmmm:
                            if isinstance(mmmmm, nn.Conv2d):
                                #print("yes")
                                torch.nn.init.xavier_normal_(mmmmm.weight, gain=1.0)
                            else:
                                for mmmmmm in mmmmm:
                                   ModelParamsInitHelper(mmmmmm, "Tanh")   
def ModelParamsInit_unetr(model):
    assert isinstance(model, nn.Module)
    for name, m in model.named_modules():
        #print(m)
        for name, mm in m.named_modules():
            for name, mmm in mm.named_modules():
                
                if isinstance(mmm, (nn.Conv2d, nn.ConvTranspose2d)):
                    ModelParamsInitHelper(mmm, "LeakyRELU")
                    #print(mmm)
                    #print("conv2D init")
                
                else:
                    for name, mmmm in mmm.named_modules():
                        if isinstance(mmmm, (monai.networks.blocks.Convolution)):
                            for mmmmm in mmmm:
                                if isinstance(mmmmm, (nn.Conv2d, nn.ConvTranspose2d)):
                                    torch.nn.init.kaiming_uniform_(mmmmm.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                                else:
                                    for mmmmmm in mmmmm:
                                        ModelParamsInitHelper(mmmmmm, "LeakyRELU")
                  
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
    #print(get_mat(LD).numpy().dtype)
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
                            LD_to = os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "U"+sub_ID, "LD", sl[0:-4])
                            HD_to = os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "U"+sub_ID, "FD", sl[0:-4])
                            PATHS = [os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "U"+sub_ID, "LD"), os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "U"+sub_ID, "FD")]
                            for PATH in PATHS:
                                if not os.path.exists(PATH):
                                    os.makedirs(PATH)
                            if(test_ssim(LD, FD)==1):
                                data = {'sub_ID': sub_ID, 'slice': sl, 'Dose': Dose, 'LDPT':LD_to, 'HDPT':HD_to}
                                df.loc[i] = data
                                i=i+1
                                export_pixel_array(LD, LD_to, os.path.join(str(Dose), sub_ID, "LD", sl[0:-4]), FD, HD_to, os.path.join(str(Dose), sub_ID, "FD", sl[0:-4]))
                            else:
                                print('fail')
                            
                            print(i)
                        #slices = []
                        
    return df
  
def find_sl_FD(FD_path, ssl):
    slices = os.listdir(FD_path)
   
    for sl in slices:
        
        if sl.split('.')[3] == ssl:
            
            return sl


def arrange_data_siemense(params, root_dir):
    
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
            sub_sub_path = [s for s in sub_sub_path if s[-1] != 'X']
 
            for scans in sub_sub_path:
                sub_ID = scans[-6:] #pt. ID
                print(sub_ID)
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
                            ssl = sl.split('.')[3]
                            LD = os.path.join(LD_path, sl)
                            sl_FD = find_sl_FD(FD_path, ssl)
                            FD = os.path.join(FD_path, sl_FD)

                            LD_to = os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "S"+sub_ID, "LD", ssl)
                            HD_to = os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "S"+sub_ID, "FD", ssl)
                            PATHS = [os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "S"+sub_ID, "LD"), os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "S"+sub_ID, "FD")]
                            for PATH in PATHS:
                                if not os.path.exists(PATH):
                                    os.makedirs(PATH)
                            if(test_ssim(LD, FD)==1):
                                data = {'sub_ID': sub_ID, 'slice': sl, 'Dose': Dose, 'LDPT':LD_to, 'HDPT':HD_to}
                                df.loc[i] = data
                                i=i+1
                               
                                export_pixel_array(LD, LD_to, os.path.join(str(Dose), sub_ID, "LD", sl[0:-4]), FD, HD_to, os.path.join(str(Dose), sub_ID, "FD", sl[0:-4]))
                            else:
                                print('fail')
                            
                            print(i)
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
    return sublists[0], sublists[1]

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
    data = torch.tensor(data)
    norm=torch.norm(data, p='fro', dim=None, keepdim=False, out=None, dtype=None)
    if(norm==0):
        norm=1
    m = torch.mean(data)
    data = (data - m)/norm
    
    return data.numpy()

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
    

def plot_result(PATH, i, results, inputs, outputs, ssim_LD_ND, ssim_RE_ND):
    #scalebar = ScaleBar(0.08, "log L2 norm", length_fraction=0.25)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    
    ax1.imshow(inputs)
    ax1.set_title('LDPT')
    
    ax2.imshow(outputs)
    ax2.set_title('NDPT')
    
    ax3.imshow(results)
    ax3.set_title('NET')
    

    f.savefig(os.path.join(PATH, "img" + str(i) + ".jpg"))
    plt.close(f)
    
def load_model(N, PATH, trainloader_2):
        net = torch.load(os.path.join(PATH, 'net.pt'))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        std = []
        fig, (ax1, ax2) = plt.subplots(2)
        
        train_loss = load_list(os.path.join(PATH, 'train_loss.pkl'))
        valid_loss = load_list(os.path.join(PATH, 'valid_loss.pkl'))
        valid_in_ssim = load_list(os.path.join(PATH, 'valid_in_ssim.pkl'))
        valid_res_ssim = load_list(os.path.join(PATH, 'valid_res_ssim.pkl'))
        print(train_loss)
        print(valid_loss)
        ax1.plot(range(0,N), train_loss, label = "training loss")
        ax1.plot(range(0,N+1), valid_loss, label = "validation loss")
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
        plt.close(fig)
        for i, data in enumerate(trainloader_2, 0):
          
            inputs = data['LDPT'].to(device)
            results = net(inputs)
           
            inputs = inputs[0,0,:,:].detach().cpu()
            #print(inputs.shape)

            outputs = data['NDPT'].to(device)
            outputs = outputs[0,0,:,:].detach().cpu() 
            results = results[0,0,:,:].detach().cpu() 
           

            if(i%100==1):
                #plot_result(np.log(results-outputs)-np.log(outputs), np.log(inputs-outputs)-np.log(outputs), np.log(results-inputs)-np.log(outputs))
                plot_result(PATH, i, results, inputs, outputs, ssim(inputs.numpy(), outputs.numpy()), ssim(results.numpy(), outputs.numpy()))
        
        #print(median(std))