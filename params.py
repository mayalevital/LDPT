#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:52:30 2021

@author: maya.fi@bm.technion.ac.il
"""

def scan_params():
    params=dict()

    params['drf'] = 50
    #params['dose_list'] = ['4', '10', '20', '50']
    params['dose_list'] = ['50']
    params['chop'] = 0
    params['real_list'] = ['A']
    params['multi_slice_n']= 1
    params['new_h'] = 128
    params['new_w']= 128  
    params['train_val_test'] = [6/50,6/50,1/100] #split of pt. between train_val_test
    params['batch_size'] = 8
    params['ker_size'] = 3
    params['encoder_depth'] = [32,32,64]
    params['center_depth'] = [64]
    params['decoder_depth'] = [64,32,32]
    #params['encoder_depth'] = [8,8,16]
    #params['center_depth'] = [16]
    #params['decoder_depth'] = [16,16,8]
    params['num_chan'] = 1
    params['num_kernels'] = 3
    params['num_of_epochs'] = 10
    params['lr'] = 0.0001
    params['momentum'] = 0.9
    params['dropout'] = 0.2
    params['net'] = 'unet'
    params['weight_decay'] = 1e-7
    params['gain'] = 1
    return params
