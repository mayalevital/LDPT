from monai.networks.nets import AutoEncoder

import os
import dataset
from dataset import ULDPT
from utilities import train_val_test_por, norm_data, plot_result, calc_ssim, ModelParamsInit, ModelParamsInit_unetr, save_run, gradient_magnitude, calc_valid_loss
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from params import scan_params
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from utilities import load_model
from utilities import arrange_data, arrange_data_siemense
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
import random
from torchsummary import summary
import time
import monai
from monai.networks.blocks import UnetBasicBlock
from monai.networks.nets import UNETR, UNet
from monai.optimizers import LearningRateFinder
import unet_2
from unet_2 import BasicUNet
from unetr import UNETR2D
import pytorch_ssim
from monai.networks.nets import AutoEncoder
plt.ioff()
torch.autograd.set_detect_anomaly(True)


def trainloaders(data):
    _dataset = ULDPT(data)
    
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

data = pd.read_pickle("./data_50.pkl", compression='infer')

t=params['t']
N = params['num_of_epochs']
l = params['lambda']
#method = params['method'][0]
N_finish = params['N_finish']
print("small weight decay, un-filtered data")
if(t==1):
    for opt in params['optimizer']:
        for network in params['net']:
            for l in params['lambda']:
                for learn in params['lr']:
                    for method in params['method']:
                        print(network)
                        print('learning rate=', learn) 
                        print('grads lambda=', l)
                        print('optimizer ', opt)
                        print('N finish=', N_finish)
                        print('method ', method)
                        
                        PATH = 'AE_' + network + '_' + str(params['num_of_epochs']) + "_epochs_" + str(learn) + "_lr_" + str(l) + "grad_loss_lambda"
                        
                        net = AutoEncoder(
                        spatial_dims=2,
                        in_channels=1,
                        out_channels=1,
                        channels=(4, 8, 16, 32),
                        strides=(1, 1, 1, 1)).to(device)

    
                        criterion = nn.L1Loss()
                        if opt == 'RMS':
                            optimizer=torch.optim.RMSprop(net.parameters(), lr=learn, alpha=0.99, eps=1e-08, weight_decay=params['weight_decay'], momentum=0, centered=False)
                        if opt == 'ADAM':
                            optimizer=torch.optim.Adam(net.parameters(), lr=learn, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
                            
                        valid_in_ssim = []
                        valid_res_ssim = []
                        [trainloader_1, trainloader_2] = trainloaders(data)
    
                        valid_loss = []
                        train_loss = []
                        for epoch in range(N):  # loop over the dataset multiple times
                            print('epoch ', epoch+1, ' out of ', N)
                            running_train_loss = 0.0
                            running_g_loss = 0.0
                            running_l1_loss = 0.0
                            SSIM_LDPT_NDPT_train = []
                            SSIM_RESU_NDPT_train = []
                            SSIM_LDPT_NDPT_valid = []
                            SSIM_RESU_NDPT_valid = []
                            net.train()
                            for i, data_train in enumerate(trainloader_1, 0):
                                
                                inputs = data_train['NDPT'].to(device)
                                outputs = data_train['NDPT'].to(device)
                                
                                optimizer.zero_grad()
                                results = net(inputs)
                                #print(results)
                                l1_train = torch.tensor(l[0])*criterion(results, outputs)
                                grad_train = torch.tensor(l[1])*criterion(gradient_magnitude(results), gradient_magnitude(outputs))
                                ssim_value = 1 - calc_ssim(results.detach().cpu(), outputs.detach().cpu())
                                if ssim_value > 0.05:
                                    #print("ssim_value")
                                    loss_train = l1_train + grad_train
                                    loss_train.backward()
                                    optimizer.step()
                                    #pull out the gradient
                                    #add the noise gaussian with alpha std
                                    running_train_loss = running_train_loss + loss_train.item()
                                    running_g_loss = running_g_loss + grad_train.item()
                                    running_l1_loss = running_l1_loss + l1_train.item()
                                    SSIM_LDPT_NDPT_train.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
                                    SSIM_RESU_NDPT_train.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
     
                            running_valid_loss = 0.0
                            running_valid_LD_loss = 0.0
                            with torch.no_grad():
                                
                                net.eval()
                                for i, data_val in enumerate(trainloader_2, 0):
                                    inputs = data_val['NDPT'].to(device)   
                                    outputs = data_val['NDPT'].to(device)
                                    #print(inputs)
                                    #print(inputs.float())
                                    results = net(inputs)
                                    l1_val = torch.tensor(l[0])*criterion(outputs, results)
                                    grad_val = torch.tensor(l[1])*criterion(gradient_magnitude(outputs), gradient_magnitude(results))
                                    l1_LDPT = torch.tensor(l[0])*criterion(data_val['LDPT'].to(device), net(data_val['LDPT'].to(device)))
                               
                                    loss_val = l1_val + grad_val      
                                    running_valid_loss = running_valid_loss + loss_val.item()
                                    running_valid_LD_loss = running_valid_LD_loss + l1_LDPT.item()
                                    SSIM_LDPT_NDPT_valid.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
                                    SSIM_RESU_NDPT_valid.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
                                                   
  
                            print("ssim valid LDPT/NDPT", np.mean(SSIM_LDPT_NDPT_valid))
                            valid_in_ssim.append(np.mean(SSIM_LDPT_NDPT_valid))
                            print("ssim valid results/NDPT", np.mean(SSIM_RESU_NDPT_valid))
                            valid_res_ssim.append(np.mean(SSIM_RESU_NDPT_valid))
                            print('[%d, %5d] training loss: %.5f' %
                                          (epoch + 1, i + 1, running_train_loss))
                            print('[%d, %5d] training grad loss: %.5f' %
                                          (epoch + 1, i + 1, running_g_loss))
                            print('[%d, %5d] training l1 loss: %.5f' %
                                          (epoch + 1, i + 1, running_l1_loss))
                            train_loss.append(running_train_loss)
                            print('[%d, %5d] validation loss: %.5f' %
                                          (epoch + 1, i + 1, running_valid_loss))
                            print('[%d, %5d] LD validation loss: %.5f' %
                                          (epoch + 1, i + 1, running_valid_LD_loss))
                            valid_loss.append(running_valid_loss)
                            
                            if N - epoch < N_finish:
                                PATH_last_models = os.path.join('AE', method, 'epoch'+str(epoch), network + '_' + str(params['num_of_epochs']) + "_epochs_" + str(learn) + "_lr_" + str(l) + "grad_loss_lambda")
                                if not os.path.exists(PATH_last_models):
                                    os.makedirs(PATH_last_models)
                                save_run(PATH_last_models, net, train_loss, valid_loss, valid_in_ssim, valid_res_ssim)
                                load_model(epoch+1, PATH_last_models, trainloader_2)   
                    
                        print('Finished Training')
                     