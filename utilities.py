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

import albumentations as A
import gdcm
from pydicom import dcmread
import numpy as np

def transforma():
    transform = A.Compose([A.Flip(p=0.5)], additional_targets={'image0': 'image', 'image1': 'image'})
    return transform

def get_mat(name):
    X = dcmread(name)
    X = X.pixel_array.astype(np.float32)
    return X

def resize_resample_images(PET_data, CT): 
    new_CT = sitk.Resample(CT, PET_data.GetSize(),
                                 sitk.Transform(), 
                                 sitk.sitkLinear,
                                 PET_data.GetOrigin(),
                                 PET_data.GetSpacing(),
                                 PET_data.GetDirection(),
                                 np.min(CT).astype('double'),
                                 CT.GetPixelID())
    return new_CT

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
                        for sl in slices:
                            LD = os.path.join(LD_path, sl)
                            FD = os.path.join(FD_path, sl)
                            data = {'sub_ID': sub_ID, 'slice': sl, 'Dose': Dose, 'LDPT':LD, 'HDPT':FD}
                            #print(data)
                            df.loc[i] = data
                            i=i+1
                            
                            print(i)
                        
    return df
                    
                        
def split(original_list, weight_list, data):
    sublists = []
    prev_index = 0
    for weight in weight_list:
        next_index = prev_index + math.ceil( (len(original_list) * weight) )

        sublists.append( original_list[prev_index : next_index] )
        prev_index = next_index
    df1 = data.iloc[sublists[0]] 
    df2 = data.iloc[sublists[1]]
    df3 = data.iloc[sublists[2]]
    df12_merge = pd.merge(df1, df2, on='sub_ID', how='inner')
    df23_merge = pd.merge(df2, df3, on='sub_ID', how='inner')
    
    df1.drop(df1.index[df1['sub_ID'] == df12_merge['sub_ID'].unique()[0]], inplace=True)
    df2.drop(df2.index[df2['sub_ID'] == df23_merge['sub_ID'].unique()[0]], inplace=True)
 
    return list(df1.index), list(df2.index), list(df3.index)                   
                
    
def train_val_test_por(params, data):
    length = len(data)
    my_list = list(range(1, length))
    weight_list = params['train_val_test'] # This equals to your 20%, 30% and 50%

    sublists = split(my_list, weight_list, data)
    #print()
    return sublists[0], sublists[1], sublists[2]

def norm_data(data):
    
    if(torch.std(data)==0):
        std=1
    else:
        std=torch.std(data)
    data = (data-torch.mean(data))/std
   

    return data





def plot_result(ct, results, inputs, outputs, mask):
    #scalebar = ScaleBar(0.08, "log L2 norm", length_fraction=0.25)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    ax1.imshow(np.log(((results-outputs)**2)/mask))
    ax1.set_title('log l2 norm % net / ND')
    #ax2.imshow(two, vmin=0, vmax=1)
    ax2.imshow(np.log(((outputs-inputs[0])**2)/mask))

    ax2.set_title('log l2 norm % LD / ND')
    #ax2.colorbar()
    ax3.imshow(np.log(((results-inputs)**2)/mask))
    ax3.set_title('log l2 norm % net / LD')
    #ax3.colorbar()
    ax4.imshow(inputs)
    ax4.set_title('LDPT')
    #ax5.imshow(inputs)
    #ax5.set_title('LDPT')
    #scalebar = ScaleBar()
    #ax1.add_artist(scalebar)
    #ax2.add_artist(scalebar)
    #ax3.add_artist(scalebar)

    plt.show()
    
def load_model(PATH, trainloader_2, loss_path):
        net = torch.load(PATH)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        std = []
        for i, data in enumerate(trainloader_2, 0):
          
            inputs = data['LDPT'].to(device)
            results = net(inputs)
            ct = inputs[0,0,:,:,0].detach().cpu()
            inputs = inputs[0,0,:,:,0].detach().cpu()
            print(inputs.shape)

            outputs = data['NDPT'].to(device)
            outputs = outputs[0,0,:,:,0].detach().cpu() 
            results = results[0,0,:,:,0].detach().cpu() 
            #print(outputs[50:60, 50:60])
            #print(inputs[50:60, 50:60])

            print("LDPT/NDPT", ssim(inputs.numpy(), outputs.numpy()))
            print("results/NDPT", ssim(results.numpy(), outputs.numpy()))
            mask = outputs
            mask[mask<0.0000001] =1
            if(i%10==1):
                #plot_result(np.log(results-outputs)-np.log(outputs), np.log(inputs-outputs)-np.log(outputs), np.log(results-inputs)-np.log(outputs))
                plot_result(ct, results, inputs, outputs, mask)
    
        #print(median(std))