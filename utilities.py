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
from skimage.metrics import structural_similarity as ssim
#from pydicom import dcmread
import numpy as np
import SimpleITK as sitk

import numpy as np
import os
from ipywidgets import interact, fixed

import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd

def resize_resample_images(PET_data, CT): 
    #print('pixel type PET: ' + str(PET_data.GetPixelIDTypeAsString()))
    #print('pixel type CT: ' + str(CT.GetPixelIDTypeAsString()))
    #print('before')
    #print('size: ' + str(CT.GetSize()))
    new_CT = sitk.Resample(CT, PET_data.GetSize(),
                                 sitk.Transform(), 
                                 sitk.sitkLinear,
                                 PET_data.GetOrigin(),
                                 PET_data.GetSpacing(),
                                 PET_data.GetDirection(),
                                 np.min(CT).astype('double'),
                                 CT.GetPixelID())
    #print('after')
    #print('size: ' + str(new_CT.GetSize()))
    return new_CT

def arrange_data(params, root_dir):
    real_list = params['real_list']
    dose_list = params['dose_list']
    full_dose = params['full_dose'][0]
    CT_folder = os.path.join(root_dir, 'CT')
    chop = params['chop']
    #print(CT_folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(CT_folder)
    reader.SetFileNames(dicom_names)
    s = 0
    CT = reader.Execute()
    df = pd.DataFrame(columns=['real', 'Dose', 'LDPT', 'HDPT', 'CT'])
    i=0
    for real in real_list:
        print(real)
        print(full_dose)
        reader = sitk.ImageFileReader()
        PET_folder = os.path.join(root_dir, real, full_dose)
        PET_DCM = os.path.join(PET_folder, os.listdir(PET_folder)[0])
        print(PET_DCM)
        reader.SetFileName(PET_DCM)
        FD_PET = reader.Execute()
        FD_PET = sitk.Cast(FD_PET, sitk.sitkUInt32)
        for dose in dose_list:
            PET_folder = os.path.join(root_dir, real, dose)
            PET_DCM = os.path.join(PET_folder, os.listdir(PET_folder)[0])
            print(PET_DCM)
            reader.SetFileName(PET_DCM)
            PET = reader.Execute()
                            
            PET = sitk.Cast(PET, sitk.sitkUInt32)
           
            if(s==0):
                new_CT = resize_resample_images(PET, CT)
                #new_CT = CT
            s = 1

            for slice in range(chop, new_CT.GetSize()[0]-chop):
                #print(np.median(sitk.GetArrayFromImage(PET)[slice, :, :]))
                d = {'real': real, 'Dose': dose, 'LDPT': [sitk.GetArrayFromImage(PET)[slice, :, :]], 'HDPT':[sitk.GetArrayFromImage(FD_PET)[slice, :, :]], 'CT':[sitk.GetArrayFromImage(new_CT)[slice, :, :]]}
                df.loc[i] = d
                i=i+1
    return df
def train_val_test_por(params):
    num_of_slices=params['num_of_slices']
    multi_slice_n=params['multi_slice_n']
    train_val_test=params['train_val_test']

    train_por = list(range(1,(num_of_slices-multi_slice_n)*train_val_test[0]))
    val_por = list(range((num_of_slices-multi_slice_n)*train_val_test[0]+1, (num_of_slices-multi_slice_n)*(train_val_test[0]+train_val_test[1])))
    test_por = list(range((num_of_slices-multi_slice_n)*(train_val_test[0]+train_val_test[1])+1, 
                      (num_of_slices-multi_slice_n)*(train_val_test[0]+train_val_test[1]+
                                                     train_val_test[2])))
    #print("train por", train_por)
    #print("val_por", val_por)
    #print("test_por", test_por)

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
    #print('before', data.dtype)
    #data1=data
    if(torch.std(data)==0):
        std=1
    else:
        std=torch.std(data)
    data = (data-torch.mean(data))/std
    #data[data1==0]=0
    #print
    #print('after norm', data.dtype)
    #print('mean', torch.mean(data))
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
    
def load_model(PATH, trainloader_2, loss_path):
        net = torch.load(PATH)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        std = []
        for i, data in enumerate(trainloader_2, 0):
          
            inputs = data['LDPT'].to(device)
            results = net(inputs)
            ct = inputs[0,0,:,:,0].detach().cpu()
            inputs = inputs[0,0,:,:,0].detach().cpu()
            print(inputs.shape)

            outputs = data['NDPT'].to(device)
            outputs = outputs[0,0,:,:,0].detach().cpu() 
            results = results[0,0,:,:,0].detach().cpu() 
            #print(outputs[50:60, 50:60])
            #print(inputs[50:60, 50:60])

            print("LDPT/NDPT", ssim(inputs.numpy(), outputs.numpy()))
            print("results/NDPT", ssim(results.numpy(), outputs.numpy()))
            mask = outputs
            mask[mask<0.0000001] =1
            if(i%10==1):
                #plot_result(np.log(results-outputs)-np.log(outputs), np.log(inputs-outputs)-np.log(outputs), np.log(results-inputs)-np.log(outputs))
                plot_result(ct, results, inputs, outputs, mask)
    
        #print(median(std))