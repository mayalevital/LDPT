#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:53:30 2021

@author: maya.fi@bm.technion.ac.il
"""

import torch.nn.functional as F
import torch.nn as nn
import torch 
import tensorflow as tf

class encoder(nn.Module):
    def __init__(self, params, num_layer):
        super(encoder, self).__init__()
        self.batch_size = params['batch_size']
        self.kernel_size = params['ker_size']
        self.multi_slice_n = params['multi_slice_n']
        self.enc_d = params['encoder_depth']
        self.pad = int((self.kernel_size-1)/2)
        self.num_chan = params['num_chan']
        self.num_kernels = params['num_kernels']
        torch.set_default_dtype(torch.float32)
        conv = nn.Conv3d
        batchnorm = nn.BatchNorm3d
        self.pool = nn.MaxPool3d((2,2,1))
        self.num_layer = num_layer
        self.enc_conv_bn = batchnorm(self.enc_d[num_layer])
        if(num_layer==0):
            self.enc_conv0 = conv(self.num_chan, self.enc_d[num_layer], kernel_size=self.kernel_size, padding=self.pad)
        else:
            self.enc_conv0 = conv(self.enc_d[num_layer-1], self.enc_d[num_layer], kernel_size=self.kernel_size, padding=self.pad)
        if(self.num_kernels>1):
            layers = []
            for i in range(0, self.num_kernels):
                layers.append(conv(self.enc_d[num_layer], self.enc_d[num_layer], kernel_size=self.kernel_size, padding=self.pad))
                layers.append(batchnorm(self.enc_d[num_layer]))
                layers.append(nn.ReLU())
            self.seq_conv = nn.Sequential(*layers)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.enc_conv0(x)
        x = self.enc_conv_bn(x)
        x = self.relu(x)
        if(self.num_kernels>1):
            x = self.seq_conv(x)
        x_fin = self.pool(x)

        return x, x_fin

class center(nn.Module):
    def __init__(self, params):
        super(center, self).__init__()
        self.batch_size = params['batch_size']
        self.kernel_size = params['ker_size']
        self.multi_slice_n = params['multi_slice_n']
        self.cen_d = params['center_depth'] 
        self.pad = int((self.kernel_size-1)/2)
        self.num_chan = params['num_chan']
        self.num_kernels = params['num_kernels']
        conv = nn.Conv3d
        batchnorm = nn.BatchNorm3d
        self.cen_conv_bn = batchnorm(self.cen_d[0])
        self.cen_conv0 = conv(self.cen_d[0], self.cen_d[0], kernel_size=self.kernel_size, padding=self.pad)
        if(self.num_kernels>1):
            layers = []
            for i in range(0, self.num_kernels):
                layers.append(conv(self.cen_d[0], self.cen_d[0], kernel_size=self.kernel_size, padding=self.pad))
                layers.append(batchnorm(self.cen_d[0]))
                layers.append(nn.ReLU())
            self.seq_conv = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        self.upsamp = nn.Upsample(scale_factor = (2,2,1), mode = 'trilinear')
        self.up_smooth = conv(self.cen_d[0], self.cen_d[0], kernel_size=self.kernel_size, padding=self.pad)
    def forward(self, x):
        x = self.relu(self.cen_conv_bn(self.cen_conv0(x)))
        if(self.num_kernels>1):
            x = self.seq_conv(x)
        x = self.up_smooth(self.upsamp(x))
        return x

class decoder(nn.Module):
    def __init__(self, params, num_layer):
        super(decoder, self).__init__()
        self.batch_size = params['batch_size']
        self.kernel_size = params['ker_size']
        self.multi_slice_n = params['multi_slice_n']
        self.cen_d = params['center_depth'] 
        self.dec_d = params['decoder_depth'] 
        self.enc_d = params['encoder_depth']
        self.pad = int((self.kernel_size-1)/2)
        self.num_chan = params['num_chan']
        self.num_kernels = params['num_kernels']
        self.num_layer = num_layer
        conv = nn.Conv3d
        batchnorm = nn.BatchNorm3d
        self.dec_conv_bn = batchnorm(self.dec_d[num_layer])
        if(num_layer==0):
            self.dec_conv0 = conv(self.enc_d[2-num_layer]+self.cen_d[num_layer], self.dec_d[num_layer], kernel_size=self.kernel_size, padding=self.pad)
        else:
            self.dec_conv0 = conv(self.enc_d[2-num_layer]+self.dec_d[num_layer-1], self.dec_d[num_layer], kernel_size=self.kernel_size, padding=self.pad)
        if(self.num_kernels>1):
            layers = []
            for i in range(0, self.num_kernels):
                layers.append(conv(self.dec_d[num_layer], self.dec_d[num_layer], kernel_size=self.kernel_size, padding=self.pad))
                layers.append(batchnorm(self.dec_d[num_layer]))
                layers.append(nn.ReLU())
            self.seq_conv = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        if(num_layer!=2):
            self.upsamp = nn.Upsample(scale_factor = (2,2,1), mode = 'trilinear')
            self.up_smooth = conv(self.dec_d[num_layer], self.dec_d[num_layer], kernel_size=self.kernel_size, padding=self.pad)
     
    def forward(self, x):
        x = self.relu(self.dec_conv_bn(self.dec_conv0(x)))
        if(self.num_kernels>1):
            x = self.seq_conv(x)
        if(self.num_layer!=2):
            x = self.up_smooth(self.upsamp(x))
        return x    
    
class Net(nn.Module):
       
    def __init__(self, params):
        super(Net, self).__init__()
        self.batch_size = params['batch_size']
        self.kernel_size = params['ker_size']
        self.multi_slice_n = params['multi_slice_n']
        self.enc_d = params['encoder_depth']
        self.cen_d = params['center_depth'] 
        self.dec_d = params['decoder_depth'] 
        self.pad = int((self.kernel_size-1)/2)
        self.num_chan = params['num_chan']
        self.num_kernels = params['num_kernels']
        conv = nn.Conv3d
        self.input_conv = conv(self.num_chan, 1, kernel_size=1)
        self.encoder0 = encoder(params, 0)
        self.encoder1 = encoder(params, 1)
        self.encoder2 = encoder(params, 2)
        self.center0 = center(params)
        self.decoder0 = decoder(params,0)
        self.decoder1 = decoder(params,1)
        self.decoder2 = decoder(params,2)
        self.fin_conv = conv(self.dec_d[2], 1, kernel_size=1)
        #self.pool = nn.MaxPool3d((1,1,self.multi_slice_n))
    def forward(self, x):
        #encoder
        #print(x[0, 0, 50:60, 50:60, 0])
        #print(x[0, 0, 50:60, 50:60, 0].float())
        #x = x.float()
        in_ = x
        
        if(self.num_chan==2):
            in_ = self.input_conv(in_)
        #print("input", in_.dtype)
        [x0, x] = self.encoder0(x)
        [x1, x] = self.encoder1(x)
        [x2, x] = self.encoder2(x)
        #center
        x = self.center0(x)
        #decoder
        x = self.decoder0(torch.cat((x2, x), dim=1))
        x = self.decoder1(torch.cat((x1, x), dim=1))
        x = self.decoder2(torch.cat((x0, x), dim=1))
        #final stage
        #print("fin_conv", self.fin_conv(x).shape)
        #print("in_", in_.shape)
        fin = self.fin_conv(x) + in_
        #fin = self.pool(fin)
        return fin



