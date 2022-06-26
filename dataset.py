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
    X = X.pixel_array.astype(np.float32)
    return X


class ULDPT(Dataset):
    """Low Dose PET dataset."""

    def __init__(self, data, root_dir, params):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.multi_slice_n = params['multi_slice_n']
        self.train_val_test = params['train_val_test']
        self.num_chan = params['num_chan']
        self.drf = params['drf']
        self.gain = params['gain']
        self.transforms = transforma()
        self.data = data
        self.length = data.size
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        #tic = time.time() 
        drf = self.drf
        data = self.data
        gain = self.gain
        LDPT = get_mat(data.iloc[idx].LDPT)
        NDPT = get_mat(data.iloc[idx].HDPT)
        #LDPT = np.array(Image.open("/tcmldrive/users/Maya/1_epochs_3_kernels_1_chan_1_slices.pt_valid__loss_ssim_.jpg")).astype(np.float32)
        #NDPT = np.array(Image.open("/tcmldrive/users/Maya/1_epochs_3_kernels_1_chan_1_slices.pt_valid__loss_ssim_.jpg")).astype(np.float32)
        Dose = data.iloc[idx].Dose
        transformed = self.transforms(image=LDPT, image0=NDPT)
        [LDPT, norm_L] = norm_data(torch.tensor(transformed["image"]))
        [NDPT, norm_N] = norm_data(torch.tensor(transformed["image0"]))
        #print(norm_L.size())
        LDPT = torch.tensor(gain)*LDPT.unsqueeze(0)
        NDPT = torch.tensor(gain)*NDPT.unsqueeze(0)
        sample = {'LDPT': LDPT, 'NDPT': NDPT, 'Dose': Dose}
                 # , 'norm_L':norm_L, 'norm_N':norm_N}
        #toc = time.time()
        #print(LDPT.type())
        return sample
    


