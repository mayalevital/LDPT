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
import unetr
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_msssim import ms_ssim
 
def save_images(df_return, out_dirs):
  
     if not os.path.exists(out_dirs):
            os.makedirs(out_dirs)
     for index, row in df_return.iterrows():
        #print(row['std'])
       
   
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    
        ax1.imshow(row['LDPT'])
        ax1.set_title('LDPT')
        
        ax2.imshow(row['NDPT'])
        ax2.set_title('NDPT')
                      #, SSIM ' + str(calc_ssim(row['LDPT'], row['NDPT'])))
        
        ax3.imshow(row['std'])
        ax3.set_title('std')
        
        ax4.imshow(row['mean'])
        ax4.set_title('mean')
                      #', SSIM '+ str(calc_ssim(row['mean'], row['NDPT'])))

        f.savefig(os.path.join(out_dirs, "img" + str(row['idx']) + ".jpg"))
        plt.close(f)
    
def dict_mean(df, i):
    df_return = pd.DataFrame(columns=['LDPT', 'NDPT', 'idx', 'mean', 'std', 'ssim_0', 'ssim_net'])
   
    mean = df['NET'].to_numpy().mean(axis=0) 
    std = df['NET'].to_numpy().std(axis=0)
  
    ssim_0 = ssim(stand_data(torch.tensor(df.iloc[0].LDPT)).numpy(), stand_data(torch.tensor(df.iloc[0].NDPT)).numpy())
    ssim_net = ssim(stand_data(torch.tensor(mean)).numpy(), stand_data(torch.tensor(df.iloc[0].NDPT)).numpy())
    data = {'LDPT': df.iloc[0].LDPT, 'NDPT': df.iloc[0].NDPT, 'idx':i, 'mean': mean, 'std':std, 'ssim_0':ssim_0, 'ssim_net':ssim_net}
    df_return.loc[0] = data
    
    return df_return

def train_test_net(trainloader_1, trainloader_2, network, N_finish, N, params, alpha, learn, method, optimizer, criterion, net, device, l, wd):
   
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    valid_in_ssim = []
    valid_res_ssim = []
             
    valid_loss = []
    train_loss = []
    
    split = np.linspace(0, len(trainloader_1), int(len(trainloader_1)/428)+1)
    split = split[1:]
   
    print(split)
    for epoch in range(N):
        print('epoch ', epoch+1, ' out of ', N)
        i=0
        iterator=iter(trainloader_1)
        for spl in split:

            running_train_loss = 0.0
            running_g_loss = 0.0
            running_l1_loss = 0.0
            SSIM_LDPT_NDPT_train = []
            SSIM_RESU_NDPT_train = []
            SSIM_LDPT_NDPT_valid = []
            SSIM_RESU_NDPT_valid = []
							
            net.train()
            while(i<=int(spl)-1):

                data_train = next(iterator)
                i=i+1
                inputs = data_train['LDPT'].to(device)
                outputs = data_train['NDPT'].to(device)
            
                optimizer.zero_grad()
                results = net(inputs)
                l1_train = torch.tensor(l[0])*criterion(results, outputs)
                grad_train = torch.tensor(l[1])*criterion(gradient_magnitude(results), gradient_magnitude(outputs)) #+ torch.tensor(l[2])*criterion(laplacian_filter(results,3), laplacian_filter(outputs,3))
                #print(scale_for_ssim(outputs).size())
                #ssim_train = torch.tensor(l[3])*(1-ms_ssim(scale_for_ssim(results), scale_for_ssim(outputs), data_range=1, size_average=False).mean())
                #print(ssim_train.item())
                ssim_value = 1 - calc_ssim(results.detach().cpu(), outputs.detach().cpu())
                if ssim_value > 0.05:
                    loss_train = l1_train + grad_train #+ ssim_train
                    loss_train.backward()
                    optimizer.step()
                    if method == 'SGLD':
                        for parameters in net.parameters():
                            parameters.grad += torch.tensor(learn*alpha).to(device, non_blocking=True)*torch.randn(parameters.grad.shape).to(device, non_blocking=True)						
                    running_train_loss = running_train_loss + loss_train.item()
                    running_g_loss = running_g_loss + grad_train.item()
                    running_l1_loss = running_l1_loss + l1_train.item()
                    SSIM_LDPT_NDPT_train.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
                    SSIM_RESU_NDPT_train.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
       
            running_valid_loss = 0.0
    		
            with torch.no_grad():
                net.eval()
                for i_t, data_val in enumerate(trainloader_2, 0):
                    inputs = data_val['LDPT'].to(device)   
                    outputs = data_val['NDPT'].to(device)
                    results = net(inputs)
                    l1_val = torch.tensor(l[0])*criterion(outputs, results)
                    grad_val = torch.tensor(l[1])*criterion(gradient_magnitude(outputs), gradient_magnitude(results)) #+ torch.tensor(l[2])*criterion(laplacian_filter(results,5), laplacian_filter(outputs,5))
                    #ssim_val = torch.tensor(l[3])*(1-ms_ssim(scale_for_ssim(results), scale_for_ssim(outputs), data_range=1, size_average=False).mean())
                    loss_val = l1_val + grad_val 
                        
                    								
                    running_valid_loss = running_valid_loss + loss_val.item()
                    SSIM_LDPT_NDPT_valid.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
                    SSIM_RESU_NDPT_valid.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
			
                scheduler.step(running_valid_loss)	
                print("learning rate = ", scheduler._last_lr)	           
                print("ssim valid LDPT/NDPT", np.mean(SSIM_LDPT_NDPT_valid))
			
                valid_in_ssim.append(np.mean(SSIM_LDPT_NDPT_valid))
			
                print("ssim valid results/NDPT", np.mean(SSIM_RESU_NDPT_valid))
                valid_res_ssim.append(np.mean(SSIM_RESU_NDPT_valid))
                print('[%d, %5d] training loss: %.5f' %
                						  (epoch + 1, i, running_train_loss))
                print('[%d, %5d] training grad loss: %.5f' %
                						  (epoch + 1, i, running_g_loss))
                print('[%d, %5d] training l1 loss: %.5f' %
                						  (epoch + 1, i, running_l1_loss))
                train_loss.append(running_train_loss)
                print('[%d, %5d] validation loss: %.5f' %
                						  (epoch + 1, i, running_valid_loss))
             
            
                valid_loss.append(running_valid_loss)
               
                if N - epoch <= N_finish:
                    PATH_last_models = os.path.join('Experiments_fin', method, network + '_' + str(params['num_of_epochs']) + "_epochs_" + str(learn) + "_lr_" + str(l) + "grad_loss_lambda" + 'weight_decay' + str(wd), 'epoch_'+str(epoch+1)+'_iter_'+str(i))
                    if not os.path.exists(PATH_last_models):
                        os.makedirs(PATH_last_models)
                    save_run(PATH_last_models, net, train_loss, valid_loss, valid_in_ssim, valid_res_ssim)
                    load_model(epoch+1, PATH_last_models, trainloader_2)   
			
               
    print('Finished Training')



def plot_im(PATH, inputs, outputs, results, i):                                
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    ax1.imshow(inputs[0,0,:,:].detach().cpu())
    ax1.set_title('LDPT')
    
    ax2.imshow(outputs[0,0,:,:].detach().cpu())
    ax2.set_title('NDPT')
    
    ax3.imshow(results[0,0,:,:].detach().cpu())
    ax3.set_title('NET')
 
    f.savefig(os.path.join(PATH, "img" + str(i) + ".jpg"))
    plt.close(f)

def dataframe_paths(PATH):
    l=[]
    df = pd.DataFrame(columns=['sub_ID', 'slice', 'Dose', 'LDPT', 'HDPT', 'scanner'])
    i=0                           
    for path, subdirs, files in os.walk(PATH):
        
        for name in files:
            LD = os.path.join(path, name)
            p = LD.split('/')
            #print(p[7])
            if p[7] == 'LD':
                FD = LD.replace('/LD/', '/FD/')
                data = {'sub_ID': p[6], 'slice': p[8], 'Dose': p[5], 'LDPT':LD, 'HDPT':FD, 'scanner': p[6][0]}
                df.loc[i] = data
                i=i+1
                #print(data)
    return df

def split_df(df, params):
    length = len(df)
    weight_list = params['train_val_test'] 
    lengths = [np.ceil(i*length) for i in weight_list]
    #h_length = [np.ceil(l/2) for l in lengths]
    
    df_U = df
    df_U_idx = df.index.to_list()
    #sublists_U = split(h_length, df_U, df_U_idx)
    sublists_U = split(lengths, df_U, df_U_idx)
    #df_S = df[df['scanner'] == 'S']
    #df_S_idx = df[df['scanner'] == 'S'].index.to_list()
    #sublists_S = split(h_length, df_S, df_S_idx)
    
     
    #return sublists_U[0]+sublists_S[0], sublists_U[1]+sublists_S[1]
    return sublists_U[0], sublists_U[1]

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

          
    
def train_val_test_por(params, data):
    length = len(data)
    #my_list = list(range(1, length))
    weight_list = params['train_val_test'] 
    lengths = [np.ceil(i*length) for i in weight_list]
    
    sublists = split(lengths, data)
    #print()
    return sublists[0], sublists[1]            


def get_mat(name):
    X = dcmread(name)
    X = X.pixel_array
    return torch.tensor(X.astype(float))            
    

def transforma():
    transform = A.Compose([A.Flip(p=0.5)], additional_targets={'image0': 'image', 'image1': 'image'})
    return transform

def export_pixel_array(in_file_LD, out_file_LD, in_file_FD, out_file_FD, dataset_name):
    LDPT = dcmread(in_file_LD).pixel_array.astype(float)
    NDPT = dcmread(in_file_FD).pixel_array.astype(float)
    transforms = transforma()
    transformed = transforms(image=LDPT, image0=NDPT)
    LDPT = transformed["image"]
    NDPT = transformed["image0"]
    #print(norm_L.size())
    LDPT = norm_data(LDPT[4:356, 4:356]).astype(np.float32)
    NDPT = norm_data(NDPT[4:356, 4:356]).astype(np.float32)
    
    h5 = h5py.File(out_file_LD)
    h5.create_dataset(dataset_name, data=LDPT)
    h5.close()
    
    h5 = h5py.File(out_file_FD)
    h5.create_dataset(dataset_name, data=NDPT)
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
    x_grad_mag = torch.sqrt(torch.tensor(1e-12)+torch.square(x_grad[0]) + torch.square(x_grad[1]))
    #x_grad_mag = torch.square(x_grad[0]) + torch.square(x_grad[1])
    return x_grad_mag

def laplacian_filter(x, kernel_size):
    #print(x.size())
    x_lap = kornia.filters.laplacian(x, kernel_size)
    #print(x_lap.size())
    return x_lap
    
def ModelParamsInitHelper(m, flag):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and flag == "LeakyRELU":
        #print("init weight LeakyRELU")
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif isinstance(m, nn.Conv2d) and flag == "Tanh":
        #print("update weight conv Tanh")
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
    #elif isinstance(m , (nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
        #print("init weight norm")
        #nn.init.constant_(m.weight, 1.0)
        #nn.init.constant_(m.bias, 0)
   

def ModelParamsInit(model):
    assert isinstance(model, nn.Module)
    for name, m in model.named_modules():
       for name, mm in m.named_modules():
            for name, mmm in mm.named_modules():
                for name, mmmm in mmm.named_modules():
                    if isinstance(mmmm, (monai.networks.blocks.Convolution)):
                        
                        for mmmmm in mmmm:
                            if isinstance(mmmmm, nn.Conv2d):
                                #print("init weight LeakyRELU")
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
                #print(mmm)
                if isinstance(mmm, (nn.Conv2d, nn.ConvTranspose2d)):
                    ModelParamsInitHelper(mmm, "LeakyRELU")
                    
                
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
                                export_pixel_array(LD, LD_to, FD, HD_to, "dataset")
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
                               
                                export_pixel_array(LD, LD_to, FD, HD_to, "dataset")
                            else:
                                print('fail')
                            
                            print(i)
                        #slices = []
                        
    return df
                        

def scale_data(data):
    
    if(torch.std(data)==0):
        std=1
    else:
        std=torch.std(data)
    
    if(torch.max(data)!=0):
     
        data = (data-torch.min(data))/(torch.max(data)-torch.min(data))-torch.tensor(0.5)
   
    return data

def scale_for_ssim(data):
    if(torch.max(data)!=0):
        data = (data-torch.min(data))/(torch.max(data)-torch.min(data))
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
        #print(train_loss)
        #print(valid_loss)
        ax1.plot(range(0,len(train_loss)), train_loss, label = "training loss")
        ax1.plot(range(0,len(valid_loss)), valid_loss, label = "validation loss")
        ax1.set(ylabel="loss")
        ax1.legend()
        ax1.set_title("training / validation loss over epochs")
        #axs[0].savefig(PATH + "train_valid_loss" + ".jpg")
        
        ax2.plot(range(0,len(valid_in_ssim)), valid_in_ssim, label = "LDPT/NDPT ssim")
        ax2.plot(range(0,len(valid_res_ssim)), valid_res_ssim, label = "result/NDPT ssim")
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
           

            if(i%10==1):
                #plot_result(np.log(results-outputs)-np.log(outputs), np.log(inputs-outputs)-np.log(outputs), np.log(results-inputs)-np.log(outputs))
                plot_result(PATH, i, results, inputs, outputs, ssim(inputs.numpy(), outputs.numpy()), ssim(results.numpy(), outputs.numpy()))
        
        #print(median(std))