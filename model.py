import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import PIL
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models


def plot(res, eval=False):
    if eval:
        pickle.dump(res, open( "evaluation_results.pkl", "wb" ) )
        plt.plot(res["train_acc_list"], c="red", label="train accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()

        plt.plot(res["test_acc_list"], c="blue", label="test accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig('linear-acc-epochs.png')
        plt.clf()

        plt.plot(res["train_loss_list"], c="red", label="train loss")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()

        plt.plot(res["test_loss_list"], c="blue", label="test loss")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig('linear-loss-epochs.png')
        plt.clf()
    else:
        pickle.dump(res, open( "feature_extract_losses.pkl", "wb" ) )
        plt.plot(res, c="red", label="train loss")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()

        plt.savefig('loss-epochs.png')
        plt.clf()
    
    


class MOCO(nn.Module):
    def __init__(self, t, m, feature_dim):
        super(MOCO, self).__init__()
        self.t = t
        self.m = m
        self.feature_dim = feature_dim
        
        self.f_k = models.resnet50()
        self.f_k.fc = nn.Sequential(   # MOCO V2 change to MLP head
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, feature_dim))
        self.f_q = models.resnet50()
        self.f_q.fc = nn.Sequential(   # MOCO V2 change to MLP head
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, feature_dim))
    
        for q_param, k_param in zip(self.f_q.parameters(), self.f_k.parameters()):
            q_param.data[:] = k_param.data[:]
            k_param.requires_grad = False
            
        
    def momentum_update(self):
        for q_param, k_param in zip(self.f_q.parameters(), self.f_k.parameters()):
            k_param.data[:] = self.m*k_param.data[:] + (1-self.m)*q_param.data[:]
            
    def forward(self, q, k, queue):
        q = F.normalize(self.f_q(q), dim=1)
        k = F.normalize(self.f_k(k), dim=1)
        N = q.shape[0]
        C = self.feature_dim
        pos_logits = torch.bmm(q.view(N,1,C), k.view(N,C,1)).squeeze(2)
        neg_logits = torch.mm(q, queue)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        
        return logits/self.t, k
    
 
       
