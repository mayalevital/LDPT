#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:49:10 2021

@author: maya.fi@bm.technion.ac.il
"""
import torch
import matplotlib.pyplot as plt
#from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np

def print_s():
    print('x',x.size())
    print('enc11',enc11.size())
    print('enc12',enc12.size())
    print('enc13',enc13.size())
    print('enc21',enc21.size())
    print('enc22',enc22.size())
    print('enc23',enc23.size())
    print('enc31',enc31.size())
    print('enc32',enc32.size())
    print('enc33',enc33.size())
    print('cen12',cen12.size())
    print('dec11',dec11.size())
    print('dec12',dec12.size())
    print('dec13',dec13.size())
    print('dec21',dec21.size())
    print('dec22',dec22.size())
    print('dec23',dec23.size())
    print('dec31',dec31.size())
    print('dec32',dec32.size())
    print('dec33',dec33.size())
    print('fin',fin.size())
    print("tanh output", F.tanh(self.fin_conv0(dec33)).size())
    print('min inputs', norm_data(data['LDPT'].double()).min())
    print('max inputs', norm_data(data['LDPT'].double()).max())
    print('min outputs', norm_data(data['NDPT'].double()).min())
    print('max outputs', norm_data(data['NDPT'].double()).max())
    print('min LDPT', data['LDPT'].double().min())
    print('max LDPT', data['LDPT'].double().max())
    print('min NDPT', data['NDPT'].double().min())
    print('max NDPT', data['NDPT'].double().max())
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
    norm = torch.norm(data,dim=(2,3))
    #print(norm.shape)
    #print(data.shape)
    for i in range(0, data.shape[0]):
        data[i, :, :, :, :]=data[i, :, :, :, :]/norm[i]
    #data=data/norm
    #data = (data-data.min())/(data.max()-data.min())
    return data

def plot_result(one, two, three):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.imshow(one, vmin=0, vmax=10)
    ax1.set_title('l2 norm % from NDPT: net output - NDPT')
    #ax1.colorbar()
    ax2.imshow(two, vmin=0, vmax=10)
    ax2.set_title(['l2 norm % from NDPT: LDPT - NDPT'])
    #ax2.colorbar()
    ax3.imshow(three, vmin=0, vmax=10)
    ax3.set_title('l2 norm % from NDPT: net output - LDPT')
    #ax3.colorbar()

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
      
        inputs = norm_data(data['LDPT'].double()).to(device)
        results = net(inputs)

        inputs = inputs[0,0,:,:,0].detach().cpu()
        outputs = norm_data(data['NDPT'].double())
        outputs = outputs[0,0,:,:,0].detach().cpu() 
        results = results[0,0,:,:,0].detach().cpu() 
        mask = outputs
        mask[mask==0.0000001] =1
        if(i%10==1):
            plot_result(100*torch.div((results-outputs)**2, mask), 100*torch.div((inputs-outputs)**2, mask), 100*torch.div((results-inputs)**2, mask))
    #print(median(std))