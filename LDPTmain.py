#!/bin/sh
#chmod +x ./LDPTmain.sh
import os
import dataset
from dataset import RAMBAM_Dataset
import u_net_torch
from u_net_torch import Net
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
import pandas as pd
from tqdm import tqdm
from utilities import load_model
from utilities import arrange_data
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
import random
from torchsummary import summary
#from matplotlib_scalebar.scalebar import ScaleBar
#from torchsummary import summary
plt.ion()


CUDA_VISIBLE_DEVICES=1 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


params = scan_params()
num_chan = params['num_chan']
root_dir = os.path.join(os.getcwd(),'RAMBAM_dataset/RAMBAM_DATASET')

#data = arrange_data(params, root_dir)
#print(data.head)
#data.to_pickle("./data.pkl", compression='infer', protocol=4)
data = pd.read_pickle("./data.pkl", compression='infer')
_dataset = RAMBAM_Dataset(data, root_dir, params)


train_por, val_por, test_por = train_val_test_por(params)

train_set = torch.utils.data.Subset(_dataset, train_por)
val_set = torch.utils.data.Subset(_dataset, val_por)
test_set = torch.utils.data.Subset(_dataset, test_por)

trainloader_1 = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
                                            shuffle=True, num_workers=8)
trainloader_2 = torch.utils.data.DataLoader(val_set, batch_size=params['batch_size'],
                                            shuffle=True, num_workers=8)
trainloader_3 = torch.utils.data.DataLoader(test_set, params['batch_size'], shuffle=True, num_workers=8)

l = []
t=2
N = params['num_of_epochs']
PATH = str(params['num_of_epochs']) + "_epochs_" + str(params['num_kernels']) + "_kernels_" + str(params['num_chan']) + "_chan_" + str(params['multi_slice_n']) + "_slices" + ".pt"
if(t==1):
    net = Net(params)
    #summary(net, input_size=(8, 1, 128, 128, 1))
    #print(net)
    ###########################!!!!!!! remember to bring it back
    net.to(device)

    criterion = nn.MSELoss()
#criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=params['lr'], momentum=params['momentum'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    writer = SummaryWriter("runs/" + PATH)
    valid_in_ssim = []
    valid_res_ssim = []
    train_loss = []
    valid_loss = []
    for epoch in range(N):  # loop over the dataset multiple times
        net.train()
        running_train_loss = 0.0
        SSIM_LDPT_NDPT_train = []
        SSIM_RESU_NDPT_train = []
        SSIM_LDPT_NDPT_valid = []
        SSIM_RESU_NDPT_valid = []
        for i, data in enumerate(trainloader_1, 0):
 
            inputs = data['LDPT'].to(device)
            outputs = data['NDPT'].to(device)
            #print(inputs[0,0,:,:,0].detach().cpu().dtype)
            optimizer.zero_grad()
            results = net(inputs)
            #results=inputs
            if(random.randint(1,100)==9):
                plt.figure()
                plt.title('right before net')
                plt.subplot(1,2,1)
                plt.imshow(results[0,0,:,:,0].detach().cpu())
                plt.subplot(1,2,2)
                plt.imshow(data['NDPT'][0,0,:,:,0])
                #print(data['NDPT'][0,0,:,:,0].dtype)
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
            inputs = data['LDPT'].to(device)   
            outputs = data['NDPT'].to(device)
            #print(inputs)
            #print(inputs.float())
            results = net(inputs)
            loss = criterion(results, outputs)        
            running_valid_loss = running_valid_loss + loss.item()
            SSIM_LDPT_NDPT_valid.append(ssim(inputs[0,0,:,:,0].detach().cpu().numpy(), outputs[0,0,:,:,0].detach().cpu().numpy()))
            SSIM_RESU_NDPT_valid.append(ssim(results[0,0,:,:,0].detach().cpu().numpy(), outputs[0,0,:,:,0].detach().cpu().numpy()))
        #writer.add_image('image validation set'+str(epoch), inputs[0,:,:,:,0].detach().cpu().numpy())        
        writer.add_scalar('training loss', running_train_loss, epoch)
        writer.add_scalar('validation loss', running_valid_loss, epoch)
        writer.add_scalar('Mean SSIM - training LDPT/NDPT', np.mean(SSIM_LDPT_NDPT_train), epoch)
        writer.add_scalar('Mean SSIM - training result/NDPT', np.mean(SSIM_RESU_NDPT_train), epoch)
        writer.add_scalar('Mean SSIM - validation LDPT/NDPT', np.mean(SSIM_LDPT_NDPT_valid), epoch)
        print("ssim valid LDPT/NDPT", np.mean(SSIM_LDPT_NDPT_valid))
        valid_in_ssim.append(np.mean(SSIM_LDPT_NDPT_valid))
        writer.add_scalar('Mean SSIM - validation result/NDPT', np.mean(SSIM_RESU_NDPT_valid), epoch)
        print("ssim valid results/NDPT", np.mean(SSIM_RESU_NDPT_valid))
        valid_res_ssim.append(np.mean(SSIM_RESU_NDPT_valid))
        print('[%d, %5d] training loss: %.3f' %
                      (epoch + 1, i + 1, running_train_loss))
        train_loss.append(running_train_loss)
        print('[%d, %5d] validation loss: %.3f' %
                      (epoch + 1, i + 1, running_valid_loss))
        valid_loss.append(running_valid_loss)
    print('Finished Training')

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(range(0,N), train_loss, label = "training loss")
    ax1.plot(range(0,N), valid_loss, label = "validation loss")
    ax1.set(ylabel="loss")
    ax1.legend()
    ax1.set_title("training / validation loss over epochs")
    #axs[0].savefig(PATH + "train_valid_loss" + ".jpg")
    
    ax2.plot(range(0,N), valid_in_ssim, label = "LDPT/NDPT ssim")
    ax2.plot(range(0,N), valid_res_ssim, label = "result/NDPT ssim")
    ax2.set(ylabel="ssim")
    ax2.legend()
    ax2.set_title("validation ssim over epochs")
    fig.savefig(PATH + "_valid__loss_ssim_" + ".jpg")
    # Specify a path
    
    
    # Save
    torch.save(net, PATH)
 
if(t==0): 
    loss_path = 'f1.txt'
    load_model(PATH, trainloader_2, loss_path)

if(t==2):
  
    for epoch in range(N):  # loop over the dataset multiple times
    
        for i, data in enumerate(trainloader_2, 0):
 
            inputs = data['LDPT'].to(device)
            outputs = data['NDPT'].to(device)
            #print(inputs.size())
  
            if(random.randint(1,100)==9):
                plt.figure()
                plt.subplot(1,2,1)
                print(data['NDPT'].size()[0])
                #m = torch.mean(data['NDPT'][0,0,:,:,0])
                #print(m)
                #plt.title(str(m))
                plt.imshow(data['LDPT'][0,0,:,:,0])
                plt.subplot(1,2,2)
                #m = torch.mean(data['LDPT'][0,0,:,:,0])
                #print(m)
                #plt.title(str(m))
                plt.imshow(data['NDPT'][0,0,:,:,0])
                m=[]
                m1=[]
                for idx in range(data['NDPT'].size()[0]):  	
                    m.append(torch.mean(data['NDPT'][idx,0,:,:,0]))
                    m1.append(torch.mean(data['LDPT'][idx,0,:,:,0]))
                print("NDPT")
                print(m)
                print("LDPT")
                print(m1)
if(t==3):
  
    for epoch in range(N):  # loop over the dataset multiple times
    
        for i, data in enumerate(trainloader_1, 0):
 
            m = []
            m1 = []
            for idx in range(data['NDPT'].size()[0]):  	
                m.append(torch.mean(data['NDPT'][idx,0,:,:,0]))
                m1.append(torch.mean(data['LDPT'][idx,0,:,:,0]))
            print("NDPT")
            print(m)
            print("LDPT")
            print(m1)
# 1. norm all data according to low dose with low dose, high dose with high dose and multiply low dose with DRF
# 2. try on 1/5 dose
# 3. display diff