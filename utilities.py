#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:49:10 2021

@author: maya.fi@bm.technion.ac.il
"""
import torch
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
from skimage.metrics import structural_similarity as ssim


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
    #using Frobenius norm as in Ultra–Low-Dose 18F-Florbetaben Amyloid PET Imaging Using Deep Learning with Multi-Contrast MRI Inputs
    #norm = torch.norm(data,dim=(2,3))
    #print(norm.shape)
    #print(data.shape)
    #for i in range(0, data.shape[0]):
    #    data[i, :, :, :, :]=data[i, :, :, :, :]/norm[i]
    #data=data/norm
    #data = (data-data.min())/(data.max()-data.min())
    #print('before norm', data.shape)
    data = (data-np.median(data))/(np.std(data))
    #print('after norm', data.shape)
    #print(data.min())
    #print(data.max())

    return data

def plot_result(ct, results, inputs, outputs, mask):
    #scalebar = ScaleBar(0.08, "log L2 norm", length_fraction=0.25)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    ax1.imshow(np.log(((results-outputs)**2)/mask))
    ax1.set_title('log l2 norm % net / ND')
    #ax2.imshow(two, vmin=0, vmax=1)
    ax2.imshow(np.log(((outputs-inputs[0])**2)/mask))

    ax2.set_title('log l2 norm % LD / ND')
    #ax2.colorbar()
    ax3.imshow(np.log(((results-inputs)**2)/mask))
    ax3.set_title('log l2 norm % net / LD')
    #ax3.colorbar()
    ax4.imshow(inputs)
    ax4.set_title('LDPT')
    #ax5.imshow(inputs)
    #ax5.set_title('LDPT')
    #scalebar = ScaleBar()
    #ax1.add_artist(scalebar)
    #ax2.add_artist(scalebar)
    #ax3.add_artist(scalebar)

    plt.show()
    
def load_model(PATH, trainloader_1, loss_path):
        net = torch.load(PATH)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        std = []
        for i, data in enumerate(trainloader_1, 0):
          
            inputs = data['LDPT'].to(device)
            results = net(inputs)
            ct = inputs[0,0,:,:,0].detach().cpu()
            inputs = inputs[0,0,:,:,0].detach().cpu()
            print(inputs.shape)

            outputs = data['NDPT'].to(device)
            outputs = outputs[0,0,:,:,0].detach().cpu() 
            results = results[0,0,:,:,0].detach().cpu() 
            print("LDPT/NDPT", ssim(inputs.numpy(), outputs.numpy()))
            print("results/NDPT", ssim(results.numpy(), outputs.numpy()))
            mask = outputs
            mask[mask<0.0000001] =1
            if(i%10==1):
                #plot_result(np.log(results-outputs)-np.log(outputs), np.log(inputs-outputs)-np.log(outputs), np.log(results-inputs)-np.log(outputs))
                plot_result(ct, results, inputs, outputs, mask)
    
        #print(median(std))