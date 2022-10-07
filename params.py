#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:52:30 2021

@author: maya.fi@bm.technion.ac.il
"""

def scan_params():
    params=dict()

    params['drf'] = 50
    #params['alpha'] = 1e-4
    params['alpha'] = 1
    params['dose_list'] = ['4', '10', '50', '20', '100']
    params['chop'] = 0
    params['real_list'] = ['A']
    params['multi_slice_n']= 1
    params['new_h'] = 128
    params['new_w']= 128  
    #params['train_val_test'] = [0.9,0.1]
    params['train_val_test'] = [0.15,0.03] #good for train
    #params['train_val_test'] = [0.12,0.03] #good for the compare
    #params['train_val_test'] = [0.71,0.15] #for dose 50 -- 50K train 10K test
    #params['train_val_test'] = [0.118,0.024]
    #params['train_val_test'] = [0.02,0.0002] #split of pt. between train_test
    params['batch_size'] = 1
    params['ker_size'] = 3
    params['iter_size'] = 5000
    params['num_chan'] = 1
    params['num_kernels'] = 3
    params['num_of_epochs'] = 20
    params['scale'] = 5e4
    params['lr'] = [5e-4]
    params['norm'] = 5e4
    params['momentum'] = 0.9
    params['dropout'] = 0.2
   
    params['net'] = ['unet']
    params['weight_decay'] = [1e-10] #best
    params['gain'] = 1
    params['t'] = 1
    params['lambda'] = [[1-0.3, 0.3]]
    params['optimizer']=['ADAM']
    params['N_finish'] = 10
    params['method']=['SGLD']
    params['compare'] = 'new'
    params['PATH'] = 'Experiment_ALL_Doses_SSIM_gate'
    params['SSIM_gate'] = 0.95
    return params
