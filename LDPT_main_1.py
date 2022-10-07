#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:21:36 2022

@author: maya.fi@bm.technion.ac.il
"""

#!/bin/sh
#chmod +x ./LDPTmain.sh
import os
#from pydicom import dcmread
from utilities import un_norm, train_test_net, train_val_test_por, ModelParamsInit, ModelParamsInit_unetr
#from SGLD_ import SGLD
import torch
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset, DataLoader, Subset
from params import scan_params
#import os
import torch.nn as nn
#import torch.optim as optim
import pandas as pd
#from tqdm import tqdm
from utilities import get_slice_ready, load_model, laplacian_filter, save_nift, results_summary, get_mat_compare
from utilities import arrange_data_old, pack_dcm, calc_res
#from torch.utils.tensorboard import SummaryWriter
#from skimage.metrics import structural_similarity as ssim
import random
#from torchsummary import summary
#import time
#import monai
#from monai.networks.blocks import UnetBasicBlock
#from monai.networks.nets import UNETR, UNet
#from monai.optimizers import LearningRateFinder
#import unet_2
from unet_2 import BasicUNet
from unetr import UNETR2D
#import pytorch_ssim
#import dataset
from dataset import ULDPT
import nibabel as nib
plt.ioff()
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

            
def trainloaders(params, data):
    _dataset = ULDPT(data, params['scale'])
    
    train_por, val_por = train_val_test_por(params, data)
    print("train portion size = ", len(train_por))
    print("test portion size = ", len(val_por))
    
    train_set = torch.utils.data.Subset(_dataset, train_por)
    val_set = torch.utils.data.Subset(_dataset, val_por)
       
    trainloader_1 = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
                                                shuffle=True, num_workers=4)
    trainloader_2 = torch.utils.data.DataLoader(val_set, batch_size=params['batch_size'],
                                                shuffle=True, num_workers=4)

    return trainloader_1, trainloader_2

CUDA_VISIBLE_DEVICES=1 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
params = scan_params()


data_ = pd.read_pickle("/tcmldrive/users/Maya/uExplorer_all_doses_data.pkl", compression='infer')
data__ = data_[data_["Dose"]!='2']
[data_train_test, data_validation] = np.array_split(data__, 2)
data_train_test_all_doses = data_train_test.sample(frac=1).reset_index(drop=True)
#data_train_test_50 = data_train_test_all_doses[data_train_test_all_doses.Dose=='50']
#print(data_train_test_50['Dose'].unique())
#print(len(data_train_test_50))
#data = data_train_test_50
data = data_train_test_all_doses
print(data['Dose'].unique())
print(len(data))
t=params['t']
N = params['num_of_epochs']
l = params['lambda']
N_finish = params['N_finish']
root = '/tcmldrive/users/Maya/'
if(t==1):
    opt = params['optimizer'][0]
    for wd in params['weight_decay']:
        for network in params['net']:
            for l in params['lambda']:
                for learn in params['lr']:
                    for method in params['method']:
                        print(network)
                        print('learning rate = ', learn) 
                        print('grads lambda = ', l)
                        print('optimizer ', opt)
                        print('N finish = ', N_finish)
                        print('method ', method)
                        print('alpha = ', params['alpha'])
                        print('weight decay = ', wd)
                        if network == 'unet':
                            #features=(32, 32, 32, 64, 128, 32)
                            if method == 'standard':
                                net = BasicUNet(spatial_dims=2, out_channels=1, features=(32, 32, 32, 64, 128, 32), norm=("group", {"num_groups": 4}), act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), dropout=params['dropout']).to(device)
                                ModelParamsInit(net)
                            if method == 'SGLD':
                                net = BasicUNet(spatial_dims=2, out_channels=1, features=(32, 32, 32, 64, 128, 32), norm=None, act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01})).to(device)
                                ModelParamsInit(net)
                           
                        if network == 'unetr':
                            net = UNETR2D(in_channels=1, out_channels=1, img_size=(352, 352), feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.2).to(device)
                            ModelParamsInit_unetr(net)
                        
                        criterion = nn.L1Loss()
                        
                        if opt == 'RMS':
                            optimizer=torch.optim.RMSprop(net.parameters(), lr=learn, alpha=0.99, eps=1e-08, weight_decay=wd, momentum=0, centered=False)
                        if opt == 'ADAM':
                            optimizer=torch.optim.Adam(net.parameters(), lr=learn, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
                        
                        [trainloader_1, trainloader_2] = trainloaders(params, data)
                        #print(len(trainloader_1))
                        train_test_net(trainloader_1, trainloader_2, network, N_finish, N, params, params['alpha'], learn, method, optimizer, criterion, net, device, l, wd, params['PATH'])


    
if(t==3):
    data_ = pd.read_pickle("/tcmldrive/users/Maya/uExplorer_all_doses_data.pkl", compression='infer')
    [data_train_test, data_validation] = np.array_split(data_, 2)
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #N_fin = [60, 80, 100, 120]
    N_fin = [1]
    params = scan_params()

    for dose in params['dose_list']:
        #data_d = data_validation[data_validation['Dose']==dose]
        data_d = data_validation.loc[data_validation['Dose'] == dose]
        [trainloader_1, trainloader_2] = trainloaders(params, data_d)
        for N_ in N_fin:
            if params['compare']=='new':
                PATH = "/tcmldrive/users/Maya/Experiments_all_doses_1109/"
            if params['compare']=='old':
                PATH = '/tcmldrive/users/Maya/Experiments_fin_/'
            results_summary(trainloader_2, N_, device, PATH, dose, params['scale'], params['compare'])

if(t==4):
    params = scan_params()
    root_dir = '/tcmldrive/databases/Public/SiemensVisionQuadra/'
    #root_dir = '/tcmldrive/databases/Public/uExplorer'
    
    df = arrange_data_old(params, root_dir, 'Siemense')
    df.to_pickle(os.path.join('/tcmldrive/users/Maya/', 'all_doses_data_test_Siemense.pkl'))
    
    
if(t==5):
    
    params = scan_params()
    root_dir_ = '/tcmldrive/users/Maya'
    #net_path = os.path.join("/tcmldrive/users/Maya/Experiments_all_doses/SGLD/unet_20_epochs_0.0005_lr_[0.7, 0.3]grad_loss_lambdaweight_decay1e-10/epoch_12_iter_10000/net.pt")
    net_path = os.path.join(root, 'Experiments_fin_/SGLD/unet_15_epochs_0.0005_lr_[0.7, 0.3, 0, 0]grad_loss_lambdaweight_decay1e-09/epoch_10_iter_2996/net.pt')

    #PATH = os.path.join(root_dir_, 'Experiments_fin_/')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #index_list = params['index_list']
    #[trainloader_1, trainloader_2] = trainloaders(params, data)
    #df = pd.DataFrame(columns=['method', 'LDPT', 'NDPT', 'idx', 'mean', 'std', 'ssim_0', 'ssim_net', 'ssim_by_iter', 'std_by_iter'])
    k=0
    input_dir = os.path.join(root_dir_, 'challange_eval/test_test/')
    pred_dir = os.path.join(root_dir_, 'challange_eval/prediction_test/')
    real_dir = os.path.join(root_dir_, 'challange_eval/ground_truth_test/')
    meta_data_path = os.path.join(root_dir_, 'meta_info_test.csv')
    list_dir = os.listdir(input_dir)
    for data_dir in list_dir:
        print(data_dir)
        LD = nib.load(os.path.join(input_dir, data_dir, os.listdir(os.path.join(input_dir, data_dir))[0])).get_fdata()
        real = nib.load(os.path.join(real_dir, data_dir, os.listdir(os.path.join(real_dir, data_dir))[0])).get_fdata()
        
        img = []
        s = LD.shape
        print(s)
                      
        l=s[2]
        for i in range(0, l):
            slice_out = real[:,:,i]
            print("____________________________")
            print('Full Dose min', slice_out.min())
            print('Full Dose max', slice_out.max())
            print("Full Dose mean", slice_out.mean())
            print("Full Dose sum", slice_out.sum())  
            
            slice_in = LD[:,:,i]
            print("____________________________")
            print('Low Dose input min', slice_in.min())
            print('Low Dose input max', slice_in.max())
            print("Low Dose input mean", slice_in.mean())
            print("Low Dose input sum", slice_in.sum())  
            inputs = get_slice_ready(device, slice_in)
            #inputs = get_slice_ready(device, slice_in, params['scale'])
            net = torch.load(net_path).to(device)
            net.eval()
            results = net(inputs)
            result = results.detach().cpu().squeeze(0).squeeze(0).numpy()
            result_fin = un_norm(result, slice_in)
            #result_fin = un_norm(result, slice_in, params['scale'])
            print("____________________________")
            print('Network prediction min', result_fin.min())
            print('Network prediction max', result_fin.max())
            print("Network prediction mean", result_fin.mean())
            print("Network prediction sum", result_fin.sum()) 
            img.append(result_fin)
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
            ax1.imshow(slice_in)
            ax1.set_title("Low dose")
            ax2.imshow(result_fin)
            ax2.set_title("Network result")
            ax3.imshow(slice_out)
            ax3.set_title("Full dose")
            f.savefig(os.path.join('/tcmldrive/users/Maya/challange_eval/', "img.jpg"))
            plt.close(f)
    
        img_fin = np.swapaxes(np.array(img), 0, 2)
        print(img_fin.shape)
        save_nift(np.array(img_fin), os.path.join(pred_dir, data_dir))

   
if t==10:
    print('hi')
    sub = 'Anonymous_ANO_20220224_1744219_121048'
    ground_truth = '/tcmldrive/databases/Public/uExplorer/uExplorerPART10/Anonymous_ANO_20220224_1744219_121048/2.886 x 600 WB NORMAL/'
    test = '/tcmldrive/databases/Public/uExplorer/uExplorerPART10/Anonymous_ANO_20220224_1744219_121048/2.886 x 600 WB D50/'
    ground_truth_out = '/tcmldrive/users/Maya/challange_eval/uExplorer_ND/'
    test_out = '/tcmldrive/users/Maya/challange_eval/uExplorer_LD/'
    pack_dcm(ground_truth, ground_truth_out, sub, 'FD')  
    pack_dcm(test, test_out, sub, '50') 
    test = '/tcmldrive/databases/Public/uExplorer/uExplorerPART10/Anonymous_ANO_20220224_1744219_121048/2.886 x 600 WB D100/'
    pack_dcm(test, test_out, sub, '100') 
    test = '/tcmldrive/databases/Public/uExplorer/uExplorerPART10/Anonymous_ANO_20220224_1744219_121048/2.886 x 600 WB D2/'
    pack_dcm(test, test_out, sub, '2') 
    
    sub = '30122021_5_20211230_170836'
    ground_truth = '/tcmldrive/databases/Public/SiemensVisionQuadra/Subject_115-117/30122021_5_20211230_170836/Full_dose/'
    test = '/tcmldrive/databases/Public/SiemensVisionQuadra/Subject_115-117/30122021_5_20211230_170836/1-50 dose/'
    ground_truth_out = '/tcmldrive/users/Maya/challange_eval/Siemense_ND/'
    test_out = '/tcmldrive/users/Maya/challange_eval/Siemense_LD/'
    pack_dcm(ground_truth, ground_truth_out, sub, 'FD')  
    pack_dcm(test, test_out, sub, '50') 
    test = '/tcmldrive/databases/Public/SiemensVisionQuadra/Subject_115-117/30122021_5_20211230_170836/1-100 dose/'
    pack_dcm(test, test_out, sub, '100') 
    test = '/tcmldrive/databases/Public/SiemensVisionQuadra/Subject_115-117/30122021_5_20211230_170836/1-2 dose/'
    pack_dcm(test, test_out, sub, '2') 

if t==11:
    ground_truth_out_S = '/tcmldrive/users/Maya/challange_eval/Siemense_ND/30122021_5_20211230_170836'
    test_out_S = '/tcmldrive/users/Maya/challange_eval/Siemense_LD/30122021_5_20211230_170836'
    ground_truth_out_U = '/tcmldrive/users/Maya/challange_eval/uExplorer_ND/Anonymous_ANO_20220224_1744219_121048'
    test_out_U = '/tcmldrive/users/Maya/challange_eval/uExplorer_LD/Anonymous_ANO_20220224_1744219_121048'
    save_to = '/tcmldrive/users/Maya/challange_eval/hist'
    FD_path_S = os.path.join(ground_truth_out_S, os.listdir(ground_truth_out_S)[0])
    FD_path_U = os.path.join(ground_truth_out_U, os.listdir(ground_truth_out_U)[0])
    i=0
    listi = ['50', '100', '2']
    n_bins = 5000
    for LD in os.listdir(test_out_S):
        
        LD_path_S = os.path.join(test_out_S, LD)
        LD_path_U = os.path.join(test_out_U, LD)
        print(LD_path_S)
        print(LD_path_U)
        FD_img_S = nib.load(FD_path_S).get_fdata()
        LD_img_S = nib.load(LD_path_S).get_fdata()
        FD_img_U = nib.load(FD_path_U).get_fdata()
        LD_img_U = nib.load(LD_path_U).get_fdata()
   

        
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        bins = np.linspace(0, 5000, 100)

        ax1.hist(FD_img_S.flatten()[FD_img_S.flatten()!=0], bins, alpha=0.5, label='FD Siemense', density=True)
        ax1.hist(LD_img_S.flatten()[LD_img_S.flatten()!=0], bins, alpha = 0.5, label='LD Siemense', density=True)
        ax1.set_title("Siemense "+ listi[i], loc='left')
        
        ax2.hist(FD_img_U.flatten()[FD_img_U.flatten()!=0], bins, alpha=0.5, label='FD uExplorer', density=True)
        ax2.hist(LD_img_U.flatten()[LD_img_U.flatten()!=0], bins, alpha=0.5, label='LD uExplorer', density=True)
        ax2.set_title("uExplorer "+ listi[i], loc='left')
        f.legend()
        f.tight_layout()
        plt.show()
        f.savefig(save_to + listi[i])
        
        plt.close(f)
        i=i+1

if t==7:
    data = pd.read_pickle('/tcmldrive/users/Maya/Experiments_fin_/50DRF60itersresults.pkl')
    B = data.groupby("method")["std"].apply(lambda x: np.median([*x], axis=0))
    
    
if(t==8):
    data = pd.read_pickle("/tcmldrive/users/Maya/all_doses_data_test_xl.pkl", compression='infer')
    HD_max_tot = []
    HD_min_tot = []
    LD_max_tot = []
    LD_min_tot = []
    print(data.Dose.unique())
    for dose in data.Dose.unique():
        data_ = data[data.Dose==dose]
        for index, row in data_.iterrows():
       
           
            LD_min_tot.append(get_mat_compare(row.LDPT).min())
            HD_min_tot.append(get_mat_compare(row.HDPT).min())
            
            LD_max_tot.append(get_mat_compare(row.LDPT).max())
            HD_max_tot.append(get_mat_compare(row.HDPT).max())
            #print(HD_sum/LD_sum)
            #print('LD min', LD_min)
            #print('LD max', LD_max)

            #print('HD min', HD_min)
            #print('HD max', HD_max)            
            
            #print('Dose', row.Dose)
#val = (2 *(val - min)/(max-min)) - 1
    print('LD min tot', np.array(LD_min_tot).min())
    print('HD min tot', np.array(HD_min_tot).min())
    print('LD max tot', np.array(LD_max_tot).max())
    print('HD max tot', np.array(HD_max_tot).max())

if(t==9):
    
    params = scan_params()
    root_dir_ = '/tcmldrive/users/Maya'
    net_path = os.path.join("/tcmldrive/users/Maya/Experiments_all_doses/SGLD/unet_20_epochs_0.0005_lr_[0.7, 0.3]grad_loss_lambdaweight_decay1e-10/epoch_12_iter_10000/net.pt")
    #PATH = os.path.join(root_dir_, 'Experiments_fin_/')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #index_list = params['index_list']
    #[trainloader_1, trainloader_2] = trainloaders(params, data)
    #df = pd.DataFrame(columns=['method', 'LDPT', 'NDPT', 'idx', 'mean', 'std', 'ssim_0', 'ssim_net', 'ssim_by_iter', 'std_by_iter'])
    k=0
    root_dir = os.path.join(root_dir_, 'challange_eval/test/')
    pred_dir = os.path.join(root_dir_, 'challange_eval/prediction/')
    meta_data_path = os.path.join(root_dir_, 'meta_info.csv')
    list_dir = os.listdir(pred_dir)
    for data_dir in list_dir:
        print(data_dir)
        if os.path.isdir(os.path.join(root_dir, data_dir)):
            print('yes')
            file_pred = os.listdir(os.path.join(pred_dir, data_dir))
            file_input = os.listdir(os.path.join(root_dir, data_dir))
            pred = nib.load(os.path.join(pred_dir, data_dir, file_pred[0])).get_fdata()
            inputs = nib.load(os.path.join(root_dir, data_dir, file_input[0])).get_fdata()

            img = []
            s = pred.shape
            print(s)
            
            l=s[2]
            print(l)
            for i in range(0, l):
                print(i)
                slice_pred = pred[:,:,i]
                print('slice pred min', slice_pred.min())
                print('slice pred max', slice_pred.max())
                print("slice pred mean", slice_pred.mean())
                print("slice pred sum", slice_pred.sum())
                slice_input = inputs[:,:,i]
                print('slice input min', slice_input.min())
                print('slice input max', slice_input.max())
                print('slice input mean', slice_input.mean())
                print("slice input sum", slice_input.sum())

                data = calc_res(slice_input, slice_input, slice_pred, 0, 0, 0)
                print(data)
                
