# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import dataset
from dataset import RIDER_Dataset
import u_net_torch
from u_net_torch import Net
from utilities import train_val_test_por
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from params import scan_params
from U_net import Unet_model
import os
import torch.nn as nn
import torch.optim as optim



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

params = scan_params()

root_dir = os.path.join(os.getcwd(), 'RIDER phantom dataset/phantom_dataset_RIDER')
_dataset = RIDER_Dataset(root_dir, params)

#for i in range(0,length):
#    sample = _dataset[i]
#    print(sample['LDPT'].shape, sample['NDPT'].shape, sample['SCCT'].shape, sample['scan_idx'], sample['z_idx'])
#split the train-val-test so that the same pahntom is not split between train and test set
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
net = Net(params)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader_1, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['LDPT']
        labels = data['NDPT']
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

