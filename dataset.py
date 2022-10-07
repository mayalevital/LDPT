#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:05:22 2021
@author: maya.fi@bm.technion.ac.il
"""
from __future__ import print_function, division
#import os
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode
from utilities import norm_data
#from utilities import stand_data
import albumentations as A
#import gdcm
from pydicom import dcmread
import numpy as np
#import time
#from PIL import Image

def transforma():
    transform = A.Compose([A.Flip(p=0.5)], additional_targets={'image0': 'image', 'image1': 'image'})
    return transform

def get_mat(name):
    X = dcmread(name)
    X = norm_data(X.pixel_array.astype(np.float32))
    return X

def get_mat_new(name):
    X = dcmread(name, force=True)
    #X = -1e-4 + X.pixel_array.astype(np.float32)*2/5e4
    X = X.pixel_array.astype(np.float32)
    return X

class ULDPT(Dataset):
    """Low Dose PET dataset."""

    def __init__(self, data, scale):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.scale = scale
        self.transforms = transforma()
        self.data = data
        self.length = data.size
        
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        #tic = time.time() 
        scale = self.scale
        data = self.data
        #print(data.iloc[idx].LDPT)
        #print('-------------------')
        #print(data.iloc[idx].HDPT)
        #print('-------------------')

        LDPT_ = get_mat_new(data.iloc[idx].LDPT)
        NDPT_ = get_mat_new(data.iloc[idx].HDPT)
     
        Dose = data.iloc[idx].Dose
        #transformed = self.transforms(image=LDPT_, image0=NDPT_)
       
       
        LDPT = torch.tensor(LDPT_/scale).unsqueeze(0)
        NDPT = torch.tensor(NDPT_/scale).unsqueeze(0)
        sample = {'LDPT': LDPT, 'NDPT': NDPT, 'Dose': Dose, 'Real': torch.tensor(NDPT_).unsqueeze(0),'LD_Real': torch.tensor(LDPT_).unsqueeze(0)}
                 # , 'norm_L':norm_L, 'norm_N':norm_N}
        #toc = time.time()
        #print(LDPT.type())
        return sample
