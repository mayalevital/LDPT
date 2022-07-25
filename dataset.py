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
from utilities import stand_data
import albumentations as A
import gdcm
from pydicom import dcmread
import numpy as np
import time
from PIL import Image

def transforma():
    transform = A.Compose([A.Flip(p=0.5)], additional_targets={'image0': 'image', 'image1': 'image'})
    return transform

def get_mat(name):
    X = dcmread(name)
    X = norm_data(X.pixel_array.astype(np.float32))
    return X

class ULDPT(Dataset):
    """Low Dose PET dataset."""

    def __init__(self, data):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        
        self.transforms = transforma()
        self.data = data
        self.length = data.size
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        #tic = time.time() 
        
        data = self.data
        
        LDPT = get_mat(data.iloc[idx].LDPT)
        NDPT = get_mat(data.iloc[idx].HDPT)
     
        Dose = data.iloc[idx].Dose
        transformed = self.transforms(image=LDPT, image0=NDPT)
       
       
        LDPT = torch.tensor(LDPT).unsqueeze(0)
        NDPT = torch.tensor(NDPT).unsqueeze(0)
        sample = {'LDPT': LDPT, 'NDPT': NDPT, 'Dose': Dose}
                 # , 'norm_L':norm_L, 'norm_N':norm_N}
        #toc = time.time()
        #print(LDPT.type())
        return sample
