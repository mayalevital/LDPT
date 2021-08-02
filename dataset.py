#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:05:22 2021

@author: maya.fi@bm.technion.ac.il
"""
from __future__ import print_function, division
import os
import torch
#import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import scipy.io as sio
#import torchvision.transforms as transforms
from skimage import data
from skimage.transform import resize as rsz
# 
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import math
import torchio as tio
import random
plt.ion()   # interactive mode
import utilities
from utilities import norm_data

def transforma():
    transforms = (tio.RandomFlip(axes=['LR', 'AP', 'IS']), tio.RandomBlur())
    trans = tio.Compose(transforms)
    
    return trans



def get_mat(self, scan_idx,z_idx, mat_name):     
    name = os.path.join(self.root_dir,"phantom"+str(scan_idx),mat_name)
    mat = sio.loadmat(name)[mat_name][:,:,z_idx:z_idx+self.multi_slice_n]
    mat = norm_data(mat)
    return mat


class RIDER_Dataset(Dataset):
    """Low Dose PET dataset."""

    def __init__(self, root_dir, params):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.length = params['length']
        self.new_h = params['new_h']
        self.new_w = params['new_w']
        self.num_of_slices = params['num_of_slices']
        self.multi_slice_n = params['multi_slice_n']
        self.train_val_test = params['train_val_test']
        self.num_chan = params['num_chan']
        self.subject = tio.datasets.FPG()
        self.transforms = transforma()
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        scan_idx = math.floor(idx/(self.num_of_slices-self.multi_slice_n))+1
        z_idx = idx%(self.num_of_slices-self.multi_slice_n)+1
        #trf = transforma(self)
        #LDPT = trf(torch.tensor(rsz(get_mat(self, scan_idx, z_idx, "LD_PT"), (self.new_h, self.new_w), anti_aliasing=True)).unsqueeze(0))
        #NDPT = trf(torch.tensor(get_mat(self, scan_idx, z_idx, "ND_PT")).unsqueeze(0))
        #SCCT = trf(torch.tensor(get_mat(self, scan_idx, z_idx, "SC_CT")).unsqueeze(0))
        LDPT = torch.tensor(rsz(get_mat(self, scan_idx, z_idx, "LD_PT"), (self.new_h, self.new_w), anti_aliasing=True)).unsqueeze(0)
        NDPT = torch.tensor(get_mat(self, scan_idx, z_idx, "ND_PT")).unsqueeze(0)
        SCCT = torch.tensor(rsz(get_mat(self, scan_idx, z_idx, "SC_CT"), (self.new_h, self.new_w), anti_aliasing=True)).unsqueeze(0)
        #print(LDPT.shape)
        seed = torch.randint(100,(1,)).data
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        LDPT = self.transforms(LDPT)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        NDPT = self.transforms(NDPT)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        SCCT = self.transforms(SCCT)
        #print(NDPT.shape)
        if(self.num_chan==1):
            sample = {'LDPT': LDPT, 'NDPT': NDPT, 'scan_idx':scan_idx, 'z_idx':z_idx}
        if(self.num_chan==2):
            sample = {'LDPT': torch.cat((LDPT, SCCT), dim=0), 'NDPT': NDPT, 'scan_idx':scan_idx, 'z_idx':z_idx}

        return sample
    


