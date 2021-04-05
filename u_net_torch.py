#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:53:30 2021

@author: maya.fi@bm.technion.ac.il
"""

import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
   
    def __init__(self, params):
        super(Net, self).__init__()
        self.batch_size = params['batch_size']
        self.kernel_size = params['ker_size']
        multi_slice_n = params['multi_slice_n']
        enc_d = params['encoder_depth']
        cen_d = params['center_depth'] 
        dec_d = params['decoder_depth'] 
        #encoder
        #stage 1
        self.enc_conv11 = nn.Conv2d(multi_slice_n, enc_d[0], kernel_size=self.kernel_size)
        self.enc_conv11_bn = nn.BatchNorm2d(enc_d[0])
        self.enc_conv12 = nn.Conv2d(enc_d[0], enc_d[0], kernel_size=self.kernel_size)
        self.enc_conv12_bn = nn.BatchNorm2d(enc_d[0])
        self.enc_conv13 = nn.Conv2d(enc_d[0], enc_d[0], kernel_size=self.kernel_size)
        self.enc_conv13_bn = nn.BatchNorm2d(enc_d[0])
        #stage 2
        self.enc_conv21 = nn.Conv2d(enc_d[0], enc_d[1], kernel_size=self.kernel_size)
        self.enc_conv21_bn = nn.BatchNorm2d(enc_d[1])
        self.enc_conv22 = nn.Conv2d(enc_d[1], enc_d[1], kernel_size=self.kernel_size)
        self.enc_conv22_bn = nn.BatchNorm2d(enc_d[1])
        self.enc_conv23 = nn.Conv2d(enc_d[1], enc_d[1], kernel_size=self.kernel_size)
        self.enc_conv23_bn = nn.BatchNorm2d(enc_d[1])
        #stage 3
        self.enc_conv31 = nn.Conv2d(enc_d[1], enc_d[2], kernel_size=self.kernel_size)
        self.enc_conv31_bn = nn.BatchNorm2d(enc_d[2])
        self.enc_conv32 = nn.Conv2d(enc_d[2], enc_d[2], kernel_size=self.kernel_size)
        self.enc_conv32_bn = nn.BatchNorm2d(enc_d[2])
        self.enc_conv33 = nn.Conv2d(enc_d[2], enc_d[2], kernel_size=self.kernel_size)
        self.enc_conv33_bn = nn.BatchNorm2d(enc_d[2])  
        #center
        self.cen_conv11 = nn.Conv2d(enc_d[2], cen_d[0], kernel_size=self.kernel_size)
        self.cen_conv11_bn = nn.BatchNorm2d(cen_d[0])
        self.cen_conv12 = nn.Conv2d(cen_d[0], cen_d[0], kernel_size=self.kernel_size)
        self.cen_conv12_bn = nn.BatchNorm2d(cen_d[0])
        self.cen_conv13 = nn.Conv2d(cen_d[0], cen_d[0], kernel_size=self.kernel_size)
        self.cen_conv13_bn = nn.BatchNorm2d(cen_d[0])
        #decoder
        #stage 1
        self.dec_conv11 = nn.Conv2d(cen_d[0], dec_d[0], kernel_size=self.kernel_size)
        self.dec_conv11_bn = nn.BatchNorm2d(dec_d[0])
        self.dec_conv12 = nn.Conv2d(dec_d[0], dec_d[0], kernel_size=self.kernel_size)
        self.dec_conv12_bn = nn.BatchNorm2d(dec_d[0])
        self.dec_conv13 = nn.Conv2d(dec_d[0], dec_d[0], kernel_size=self.kernel_size)
        self.dec_conv13_bn = nn.BatchNorm2d(dec_d[0])
        #stage 2
        self.dec_conv21 = nn.Conv2d(dec_d[0], dec_d[1], kernel_size=self.kernel_size)
        self.dec_conv21_bn = nn.BatchNorm2d(dec_d[1])
        self.dec_conv22 = nn.Conv2d(dec_d[1], dec_d[1], kernel_size=self.kernel_size)
        self.dec_conv22_bn = nn.BatchNorm2d(dec_d[1])
        self.dec_conv23 = nn.Conv2d(dec_d[1], dec_d[1], kernel_size=self.kernel_size)
        self.dec_conv23_bn = nn.BatchNorm2d(dec_d[1])
        #stage 3
        self.dec_conv31 = nn.Conv2d(dec_d[1], dec_d[2], kernel_size=self.kernel_size)
        self.dec_conv31_bn = nn.BatchNorm2d(dec_d[2])
        self.dec_conv32 = nn.Conv2d(dec_d[2], dec_d[2], kernel_size=self.kernel_size)
        self.dec_conv32_bn = nn.BatchNorm2d(dec_d[2])
        self.dec_conv33 = nn.Conv2d(dec_d[2], dec_d[2], kernel_size=self.kernel_size)
        self.dec_conv33_bn = nn.BatchNorm2d(dec_d[2])
        #final
        self.fin_conv0 = nn.Conv2d(dec_d[2], 1, kernel_size=1)
        #other operations
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.UpsamplingBilinear2d(size=None, scale_factor=2)
    def forward(self, x):
        #encoder
        #stage1
        x = F.relu(self.enc_conv11_bn(self.enc_conv11(x)))
        x = F.relu(self.enc_conv12_bn(self.enc_conv12(x)))
        x = self.pool(F.relu(self.enc_conv13_bn(self.enc_conv13(x))))
        #stage 2
        x = F.relu(self.enc_conv21_bn(self.enc_conv21(x)))
        x = F.relu(self.enc_conv22_bn(self.enc_conv22(x)))
        x = self.pool(F.relu(self.enc_conv23_bn(self.enc_conv23(x))))
        #stage3
        x = F.relu(self.enc_conv31_bn(self.enc_conv31(x)))
        x = F.relu(self.enc_conv32_bn(self.enc_conv32(x)))
        x = self.pool(F.relu(self.enc_conv33_bn(self.enc_conv33(x))))
        #center
        x = F.relu(self.cen_conv11_bn(self.cen_conv11(x)))
        x = F.relu(self.cen_conv12_bn(self.cen_conv12(x)))
        x = F.relu(self.cen_conv13_bn(self.cen_conv13(x)))
        #decoder
        #stage1
        x = F.relu(self.dec_conv11_bn(self.dec_conv11(x)))
        x = F.relu(self.dec_conv12_bn(self.dec_conv12(x)))
        x = self.up(F.relu(self.dec_conv13_bn(self.dec_conv13(x))))
        #stage 2
        x = F.relu(self.dec_conv21_bn(self.dec_conv21(x)))
        x = F.relu(self.dec_conv22_bn(self.dec_conv22(x)))
        x = self.up(F.relu(self.dec_conv23_bn(self.dec_conv23(x))))
        #stage3
        x = F.relu(self.dec_conv31_bn(self.dec_conv31(x)))
        x = F.relu(self.dec_conv32_bn(self.dec_conv32(x)))
        x = self.up(F.relu(self.dec_conv33_bn(self.dec_conv33(x))))  
        #final
        x = self.fin_conv0(x)
        return x



