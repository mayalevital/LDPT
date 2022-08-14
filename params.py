#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:52:30 2021

@author: maya.fi@bm.technion.ac.il
"""

def scan_params():
    params=dict()

    params['drf'] = 100
    #params['alpha'] = 1e-6
    params['alpha'] = 1e-4
    #params['dose_list'] = ['4', '10', '20', '50']
    #params['dose_list'] = ['4', '10', '20', '50', '100']
    params['dose_list'] = ['50']
    params['chop'] = 0
    params['real_list'] = ['A']
    params['multi_slice_n']= 1
    params['new_h'] = 128
    params['new_w']= 128  
    #params['train_val_test'] = [0.2,0.05] #split of pt. between train_test
    params['train_val_test'] = [0.2,0.05] #split for debugging
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
    params['num_of_epochs'] = 15
    params['lr'] = [5e-4]
    #params['lr'] = [5e-4] good for unets
    #params['lr'] = [5e-4]
    
    params['momentum'] = 0.9
    params['dropout'] = 0.2
   
    params['net'] = ['unet']
    #params['weight_decay'] = 1e-12
    params['weight_decay'] = [1e-11] #best
    params['gain'] = 1
    params['t'] = 1
    #params['lambda'] = [[1-0.3-0.01, 0.3, 0, 0.01]]
    params['lambda'] = [[1-0.3, 0.3, 0, 0]]
    #params['lambda'] = [[1-1e-2, 1e-2], [1-1e-4,1e-4], [1,0], [1/2, 1/2]]
    params['optimizer']=['ADAM']
    params['N_finish'] = 5
    params['method']=['SGLDs']
    params['index_list'] = [811, 821]
    return params
#[1-4e-1, 3e-1, 1e-1], 