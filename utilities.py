#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:49:10 2021

@author: maya.fi@bm.technion.ac.il
"""
def train_val_test_por(params):
    num_of_slices=params['num_of_slices']
    multi_slice_n=params['multi_slice_n']
    train_val_test=params['train_val_test']

    train_por = list(range(1,(num_of_slices-multi_slice_n)*train_val_test[0]))
    val_por = list(range((num_of_slices-multi_slice_n)*train_val_test[0]+1, (num_of_slices-multi_slice_n)*(train_val_test[0]+train_val_test[1])))
    test_por = list(range((num_of_slices-multi_slice_n)*(train_val_test[0]+train_val_test[1])+1, 
                      (num_of_slices-multi_slice_n)*(train_val_test[0]+train_val_test[1]+
                                                     train_val_test[2])))
    return train_por, val_por, test_por

