#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:05:22 2021

@author: maya.fi@bm.technion.ac.il
"""
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode
from utilities import norm_data
import albumentations as A

#from pydicom import dcmread

def transforma():
    #transform = A.Compose([A.Flip(p=0.5), A.Blur(p=0.5)], additional_targets={'image0': 'image', 'image1': 'image'})
    transform = A.Compose([A.Flip(p=0.5)], additional_targets={'image0': 'image', 'image1': 'image'})
    return transform



def get_mat(self, scan_idx,z_idx, mat_name):     
    name = os.path.join(self.root_dir,"Reframed_"+str(scan_idx),mat_name)
    X = dcmread(name)
    X = X.pixel_array
    mat = X[:,:,z_idx:z_idx+self.multi_slice_n]
    mat = norm_data(mat)
    return mat


class RAMBAM_Dataset(Dataset):
    """Low Dose PET dataset."""

    def __init__(self, data, root_dir, params):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.length = params['length']

        self.num_of_slices = params['num_of_slices']
        self.multi_slice_n = params['multi_slice_n']
        self.train_val_test = params['train_val_test']
        self.num_chan = params['num_chan']
        self.drf = params['drf']
        self.transforms = transforma()
        self.data = data
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        drf = self.drf
        data = self.data
  
        LDPT = data.iloc[idx].LDPT[0].astype(float)
        #print('before trans ', LDPT.dtype)

        #print('after', LDPT[60:65, 60:65])
        #norm_data(LDPT)
        NDPT = data.iloc[idx].HDPT[0].astype(float)
        SCCT = data.iloc[idx].CT[0].astype(float)
        Dose = data.iloc[idx].Dose
        transformed = self.transforms(image=LDPT, image0=NDPT, image1=SCCT)
        #print(LDPT.shape)
        LDPT = transformed["image"]
        NDPT = transformed["image0"]
        SCCT = transformed["image1"]
        
        #LDPT[LDPT<1]=0
        #print('after trans ', LDPT.dtype)
        #LDPT = norm_data(torch.tensor(LDPT)).unsqueeze(0).unsqueeze(3)
        #NDPT = norm_data(torch.tensor(NDPT)).unsqueeze(0).unsqueeze(3)
        #SCCT = norm_data(torch.tensor(SCCT)).unsqueeze(0).unsqueeze(3)
        LDPT = norm_data(drf*torch.tensor(LDPT)).unsqueeze(0).unsqueeze(3)
        NDPT = norm_data(torch.tensor(NDPT)).unsqueeze(0).unsqueeze(3)
        SCCT = norm_data(torch.tensor(SCCT)).unsqueeze(0).unsqueeze(3)
        #print('after norm ', LDPT.dtype)


        
        #print(LDPT)
        #print(NDPT.shape)
        if(self.num_chan==1):
            sample = {'LDPT': LDPT, 'NDPT': NDPT, 'Dose': Dose}
        if(self.num_chan==2):
            sample = {'LDPT': torch.cat((LDPT, SCCT), dim=0), 'NDPT': NDPT, 'Dose': Dose}

        return sample
    


