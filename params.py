#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:52:30 2021

@author: maya.fi@bm.technion.ac.il
"""

def scan_params():
    params=dict()

    #params['real_list'] = ['A', 'C', 'D', 'F']
    #params['dose_list;] = ['1s', '2s', '4s', '8s', '16s']
    params['real_list'] = ['F', 'D', 'C', 'A', 'C']
    #params['real_list'] = ['A']
    params['drf'] = 4
    params['dose_list'] = ['16s']
    params['full_dose'] = ['64s']
    params['chop'] = 28

    params['multi_slice_n']= 1
    params['num_of_slices'] = 128 - params['chop']*2
    params['num_scans'] = 3
    params['length'] = params['num_of_slices']*len(params['real_list'])*len(params['dose_list'])
    params['new_h'] = 128
    params['new_w']= 128  
    params['train_val_test'] = [2,2,1] #split of pt. between train_val_test
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
    params['num_of_epochs'] = 50
    params['lr'] = 0.0001
    params['momentum'] = 0.9
    return params
