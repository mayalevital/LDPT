#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 22:16:17 2022

@author: maya.fi@bm.technion.ac.il
"""

"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.
"""
import numpy as np
import os
from unet_2 import BasicUNet
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from params import scan_params
from utilities import train_val_test_por, ModelParamsInit, calc_res, print_function
from dataset import ULDPT
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utilities import gradient_magnitude, compute_ssim
torch.set_default_dtype(torch.float32)

CUDA_VISIBLE_DEVICES=1 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 1

DIR = os.getcwd()
EPOCHS = 10


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


def define_model(trial):
    method = 'SGLD'
    if method == 'standard':
        net = BasicUNet(spatial_dims=2, out_channels=1, features=(32, 32, 32, 64, 128, 32), norm=("group", {"num_groups": 4}), act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), dropout=params['dropout']).to(device)
        ModelParamsInit(net)
    if method == 'SGLD':
        net = BasicUNet(spatial_dims=2, out_channels=1, features=(32, 32, 32, 64, 128, 32), norm=None, act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01})).to(device)
        ModelParamsInit(net)
              
    return net        
                       
                 

def objective(trial):

    method = 'SGLD'
    net = define_model(trial).to(DEVICE)
    l = [0.7, 0.3]
    # Generate the optimizers.
    lr = 5e-4
    wd = trial.suggest_float("wd", 1e-12, 1e-9)
    alpha = trial.suggest_float("alpha", 1e-2, 1)
    optimizer=torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.L1Loss()
    split = np.linspace(0, len(trainloader_1), int(len(trainloader_1)/params['iter_size'])+1)
    split = split[1:]
    
    for epoch in range(EPOCHS):
        print('epoch ', epoch+1, ' out of ', EPOCHS)
        i=0
        print("SSIM range", params['SSIM_gate'] + 0.01*epoch)
        iterator=iter(trainloader_1)
        for spl in split:
            
            net.train()
            while(i<=int(spl)-1):

                data_train = next(iterator)
                i=i+1
                inputs = data_train['LDPT'].to(device)
                outputs = data_train['NDPT'].to(device)
                optimizer.zero_grad()
                results = net(inputs)
                l1_train = torch.tensor(l[0])*criterion(results, outputs)
                grad_train = torch.tensor(l[1])*criterion(gradient_magnitude(results), gradient_magnitude(outputs))
                ssim_value = compute_ssim(results.squeeze(0).squeeze(0).detach().cpu().numpy(), outputs.squeeze(0).squeeze(0).detach().cpu().numpy())
                if ssim_value < params['SSIM_gate'] + 0.01*epoch :
                    loss_train = l1_train + grad_train
                    loss_train.backward()
                    optimizer.step()
                    if method == 'SGLD':
                        for parameters in net.parameters():
                            parameters.grad += torch.tensor(lr*alpha).to(device, non_blocking=True)*torch.randn(parameters.grad.shape).to(device, non_blocking=True)						
            
            SSIM_RESU_NDPT_valid = pd.DataFrame(columns=['epoch', 'iter','Dose', 'SSIM0', 'SSIM','PSNR','NRMSE'])
            running_valid_loss = 0.0
            k_v=0                   
            with torch.no_grad():
                net.eval()
                for i_t, data_val in enumerate(trainloader_2, 0):
                    inputs = data_val['LDPT'].to(device)   
                    outputs = data_val['NDPT'].to(device)
                    results = net(inputs)
     
                    l1_val = torch.tensor(l[0])*criterion(outputs, results)
                    grad_val = torch.tensor(l[1])*criterion(gradient_magnitude(outputs), gradient_magnitude(results)) 
                    
                    loss_val = l1_val + grad_val
                    running_valid_loss = running_valid_loss + loss_val.item()
                    if outputs.detach().cpu().max()!=0 and inputs.detach().cpu().max()!=0:
                        SSIM_RESU_NDPT_valid_data = calc_res(inputs.squeeze(0).squeeze(0).detach().cpu().numpy(), outputs.squeeze(0).squeeze(0).detach().cpu().numpy(), results.squeeze(0).squeeze(0).detach().cpu().numpy(), data_val['Dose'][0], epoch+1, i)
                        SSIM_RESU_NDPT_valid.loc[k_v] = SSIM_RESU_NDPT_valid_data
                        k_v=k_v+1  
                        
            scheduler.step(running_valid_loss)	
            print("valid results:")
            [valid_in_ssim_, valid_res_ssim_] = print_function(SSIM_RESU_NDPT_valid)
        
            trial.report(valid_res_ssim_, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return valid_res_ssim_


if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES=1 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    params = scan_params()


    data_ = pd.read_pickle("/tcmldrive/users/Maya/uExplorer_all_doses_data.pkl", compression='infer')
    data__ = data_[data_["Dose"]!='2']
    [data_train_test, data_validation] = np.array_split(data__, 2)
    data_train_test_all_doses = data_train_test.sample(frac=1).reset_index(drop=True)
    data = data_train_test_all_doses
    print(data['Dose'].unique())
    print(len(data))
    t=params['t']
    criterion = nn.L1Loss()
    
                        
    [trainloader_1, trainloader_2] = trainloaders(params, data)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))