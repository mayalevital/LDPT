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

import gdcm
from pydicom import dcmread
import numpy as np
import time
from PIL import Image
import h5py





def get_mat(name):
    X = h5py.File(name, 'r')
    
    return X["dataset"].value


class ULDPT(Dataset):
    """Low Dose PET dataset."""

    def __init__(self, length, data):
        
        self.length = length
        self.data = data
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        data = self.data
        LDPT = torch.tensor(get_mat(data.iloc[idx].LDPT)).unsqueeze(0)
        NDPT = torch.tensor(get_mat(data.iloc[idx].HDPT)).unsqueeze(0)

        sample = {'LDPT': LDPT, 'NDPT': NDPT}

        return sample
    


