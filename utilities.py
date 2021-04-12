#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:49:10 2021

@author: maya.fi@bm.technion.ac.il
"""
import torch
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

def norm_data(data):
    norm = torch.unsqueeze(torch.norm(data,dim=(2,3)),dim=2)
    norm = torch.unsqueeze(norm,dim=3)
    data=data/norm
    return data

def plot_result(results, inputs, outputs):
    res = results.clone()
    res = res[0,1,:,:].detach().numpy()
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.imshow(inputs[0,1,:,:])
    ax1.set_title('LDPT')
    ax2.imshow(outputs[0,1,:,:])
    ax2.set_title('NDPT')
    ax3.imshow(res)
    ax3.set_title('net output')
    plt.show()
    
