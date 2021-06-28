#!/bin/sh
#chmod +x ./LDPTmain.sh
import os
import dataset
from dataset import RIDER_Dataset
import u_net_torch_1
from u_net_torch_1 import Net
from utilities import train_val_test_por
from utilities import norm_data
from utilities import plot_result
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from params import scan_params
import os
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utilities import load_model
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim

CUDA_VISIBLE_DEVICES=1 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


params = scan_params()

root_dir = os.path.join(os.getcwd(), 'RIDER phantom dataset/phantom_dataset_RIDER')
_dataset = RIDER_Dataset(root_dir, params)


train_por, val_por, test_por = train_val_test_por(params)

train_set = torch.utils.data.Subset(_dataset, train_por)
val_set = torch.utils.data.Subset(_dataset, val_por)
test_set = torch.utils.data.Subset(_dataset, test_por)

trainloader_1 = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
                                            shuffle=True, num_workers=8)
trainloader_2 = torch.utils.data.DataLoader(val_set, batch_size=params['batch_size'],
                                            shuffle=True, num_workers=8)
trainloader_3 = torch.utils.data.DataLoader(test_set, params['batch_size'],
                                            shuffle=True, num_workers=8)
net = Net(params).double()
net.to(device)

criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.0002, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
l = []
t=1
if(t==1):
    writer = SummaryWriter('runs/300_epochs_2e_4_lr_0_9_momentum_norm_frov_1_conv_layer')
    for epoch in range(300):  # loop over the dataset multiple times
        net.train()
        running_train_loss = 0.0
        SSIM_LDPT_NDPT_train = []
        SSIM_RESU_NDPT_train = []
        SSIM_LDPT_NDPT_valid = []
        SSIM_RESU_NDPT_valid = []
        for i, data in enumerate(trainloader_1, 0):
            inputs = norm_data(data['LDPT'].double()).to(device)      
            outputs = norm_data(data['NDPT'].double()).to(device)  
            optimizer.zero_grad()
            results = net(inputs)
            loss = criterion(results, outputs)
            loss.backward()
            optimizer.step()
            running_train_loss = running_train_loss + loss.item()
            SSIM_LDPT_NDPT_train.append(ssim(inputs[0,0,:,:,0].detach().cpu().numpy(), outputs[0,0,:,:,0].detach().cpu().numpy()))
            SSIM_RESU_NDPT_train.append(ssim(results[0,0,:,:,0].detach().cpu().numpy(), outputs[0,0,:,:,0].detach().cpu().numpy()))
        scheduler.step(running_train_loss)
        net.eval()     # Optional when not using Model Specific layer
        running_valid_loss = 0.0
        for i, data in enumerate(trainloader_2, 0):
            inputs = norm_data(data['LDPT'].double()).to(device)      
            outputs = norm_data(data['NDPT'].double()).to(device)          
            results = net(inputs)
            loss = criterion(results, outputs)        
            running_valid_loss = running_valid_loss + loss.item()
            SSIM_LDPT_NDPT_valid.append(ssim(inputs[0,0,:,:,0].detach().cpu().numpy(), outputs[0,0,:,:,0].detach().cpu().numpy()))
            SSIM_RESU_NDPT_valid.append(ssim(results[0,0,:,:,0].detach().cpu().numpy(), outputs[0,0,:,:,0].detach().cpu().numpy()))
        writer.add_image('image validation set'+str(epoch), inputs[0,:,:,:,0].detach().cpu().numpy())        
        writer.add_scalar('training loss', running_train_loss, epoch)
        writer.add_scalar('validation loss', running_valid_loss, epoch)
        writer.add_scalar('Mean SSIM - training LDPT/NDPT', np.mean(SSIM_LDPT_NDPT_train), epoch)
        writer.add_scalar('Mean SSIM - training result/NDPT', np.mean(SSIM_RESU_NDPT_train), epoch)
        writer.add_scalar('Mean SSIM - validation LDPT/NDPT', np.mean(SSIM_LDPT_NDPT_valid), epoch)
        writer.add_scalar('Mean SSIM - validation result/NDPT', np.mean(SSIM_RESU_NDPT_valid), epoch)
        print('[%d, %5d] training loss: %.3f' %
                      (epoch + 1, i + 1, running_train_loss))
        print('[%d, %5d] validation loss: %.3f' %
                      (epoch + 1, i + 1, running_valid_loss))
    print('Finished Training')

    # Specify a path
    PATH = "300_epochs_2e_4_lr_0_9_momentum_norm_frov_1_conv_layer.pt"
    
    # Save
    torch.save(net, PATH)
 
if(t==0): 
    PATH = "300_epochs_2e_4_lr_0_9_momentum_norm_frov_1_conv_layer.pt"
    loss_path = 'f1.txt'
    load_model(PATH, trainloader_1, loss_path)


# 1. norm all data according to low dose with low dose, high dose with high dose and multiply low dose with DRF
# 2. try on 1/5 dose
# 3. display diff