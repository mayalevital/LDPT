#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:52:30 2021

@author: maya.fi@bm.technion.ac.il
"""

def scan_params():
    params=dict()
    params['multi_slice_n']= 1
    params['num_of_slices'] = 47
    params['num_scans'] = 9
    params['length'] = (params['num_of_slices']-params['multi_slice_n'])*params['num_scans']
    params['new_h'] = 128
    params['new_w']= 128  
    params['train_val_test'] = [5,2,2] #split of pt. between train_val_test
    params['batch_size'] = 8
    params['ker_size'] = 3
    params['encoder_depth'] = [32,32,64]
    params['center_depth'] = [64]
    params['decoder_depth'] = [64,32,32]
    params['num_chan'] = 1

    return params
