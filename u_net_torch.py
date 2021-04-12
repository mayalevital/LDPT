#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:53:30 2021

@author: maya.fi@bm.technion.ac.il
"""

import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    
    def print_dims():
        print('x',x.size())
        print('enc11',enc11.size())
        print('enc12',enc12.size())
        print('enc13',enc13.size())
        print('enc21',enc21.size())
        print('enc22',enc22.size())
        print('enc23',enc23.size())
        print('enc31',enc31.size())
        print('enc32',enc32.size())
        print('enc33',enc33.size())
        print('dec11',dec11.size())
        print('dec12',dec12.size())
        print('dec13',dec13.size())
        print('dec21',dec21.size())
        print('dec22',dec22.size())
        print('dec23',dec23.size())
        print('dec31',dec31.size())
        print('dec32',dec32.size())
        print('dec33',dec33.size())
        print('fin',fin.size())
        
    def __init__(self, params):
        super(Net, self).__init__()
        self.batch_size = params['batch_size']
        self.kernel_size = params['ker_size']
        multi_slice_n = params['multi_slice_n']
        enc_d = params['encoder_depth']
        cen_d = params['center_depth'] 
        dec_d = params['decoder_depth'] 
        pad = int((self.kernel_size-1)/2)
        #encoder
        #stage 1
        self.enc_conv11 = nn.Conv2d(multi_slice_n, enc_d[0], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv11_bn = nn.BatchNorm2d(enc_d[0])
        self.enc_conv12 = nn.Conv2d(enc_d[0], enc_d[0], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv12_bn = nn.BatchNorm2d(enc_d[0])
        self.enc_conv13 = nn.Conv2d(enc_d[0], enc_d[0], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv13_bn = nn.BatchNorm2d(enc_d[0])
        #stage 2
        self.enc_conv21 = nn.Conv2d(enc_d[0], enc_d[1], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv21_bn = nn.BatchNorm2d(enc_d[1])
        self.enc_conv22 = nn.Conv2d(enc_d[1], enc_d[1], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv22_bn = nn.BatchNorm2d(enc_d[1])
        self.enc_conv23 = nn.Conv2d(enc_d[1], enc_d[1], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv23_bn = nn.BatchNorm2d(enc_d[1])
        #stage 3
        self.enc_conv31 = nn.Conv2d(enc_d[1], enc_d[2], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv31_bn = nn.BatchNorm2d(enc_d[2])
        self.enc_conv32 = nn.Conv2d(enc_d[2], enc_d[2], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv32_bn = nn.BatchNorm2d(enc_d[2])
        self.enc_conv33 = nn.Conv2d(enc_d[2], enc_d[2], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv33_bn = nn.BatchNorm2d(enc_d[2])  
        #center
        self.cen_conv11 = nn.Conv2d(enc_d[2], cen_d[0], kernel_size=self.kernel_size, padding=pad)
        self.cen_conv11_bn = nn.BatchNorm2d(cen_d[0])
        self.cen_conv12 = nn.Conv2d(cen_d[0], cen_d[0], kernel_size=self.kernel_size, padding=pad)
        self.cen_conv12_bn = nn.BatchNorm2d(cen_d[0])
        self.cen_conv13 = nn.Conv2d(cen_d[0], cen_d[0], kernel_size=self.kernel_size, padding=pad)
        self.cen_conv13_bn = nn.BatchNorm2d(cen_d[0])
        #decoder
        #stage 1
        self.dec_conv11 = nn.Conv2d(cen_d[0], dec_d[0], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv11_bn = nn.BatchNorm2d(dec_d[0])
        self.dec_conv12 = nn.Conv2d(dec_d[0], dec_d[0], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv12_bn = nn.BatchNorm2d(dec_d[0])
        self.dec_conv13 = nn.Conv2d(dec_d[0], dec_d[0], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv13_bn = nn.BatchNorm2d(dec_d[0])
        #stage 2
        self.dec_conv21 = nn.Conv2d(dec_d[0], dec_d[1], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv21_bn = nn.BatchNorm2d(dec_d[1])
        self.dec_conv22 = nn.Conv2d(dec_d[1], dec_d[1], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv22_bn = nn.BatchNorm2d(dec_d[1])
        self.dec_conv23 = nn.Conv2d(dec_d[1], dec_d[1], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv23_bn = nn.BatchNorm2d(dec_d[1])
        #stage 3
        self.dec_conv31 = nn.Conv2d(dec_d[1], dec_d[2], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv31_bn = nn.BatchNorm2d(dec_d[2])
        self.dec_conv32 = nn.Conv2d(dec_d[2], dec_d[2], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv32_bn = nn.BatchNorm2d(dec_d[2])
        self.dec_conv33 = nn.Conv2d(dec_d[2], dec_d[2], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv33_bn = nn.BatchNorm2d(dec_d[2])
        #final
        self.fin_conv0 = nn.Conv2d(dec_d[2], multi_slice_n, kernel_size=1)
        #other operations
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.UpsamplingBilinear2d(size=None, scale_factor=2)
    def forward(self, x):
        #encoder
        #stage1
        enc11 = F.relu(self.enc_conv11_bn(self.enc_conv11(x)))
        enc12 = F.relu(self.enc_conv12_bn(self.enc_conv12(enc11)))
        enc13 = self.pool(F.relu(self.enc_conv13_bn(self.enc_conv13(enc12))))
        #stage 2
        enc21 = F.relu(self.enc_conv21_bn(self.enc_conv21(enc13)))
        enc22 = F.relu(self.enc_conv22_bn(self.enc_conv22(enc21)))
        enc23 = self.pool(F.relu(self.enc_conv23_bn(self.enc_conv23(enc22))))
        #stage3
        enc31 = F.relu(self.enc_conv31_bn(self.enc_conv31(enc23)))
        enc32 = F.relu(self.enc_conv32_bn(self.enc_conv32(enc31)))
        enc33 = self.pool(F.relu(self.enc_conv33_bn(self.enc_conv33(enc32))))
        #center
        cen11 = F.relu(self.cen_conv11_bn(self.cen_conv11(enc33)))
        cen12 = F.relu(self.cen_conv12_bn(self.cen_conv12(cen11)))
        cen13 = F.relu(self.cen_conv13_bn(self.cen_conv13(cen12)))
        #decoder
        #stage1
     

        dec11 = F.relu(self.dec_conv11_bn(self.dec_conv11(cen13+enc33)))
        dec12 = F.relu(self.dec_conv12_bn(self.dec_conv12(dec11)))
        dec13 = self.up(F.relu(self.dec_conv13_bn(self.dec_conv13(dec12))))
        #stage 2
        dec21 = F.relu(self.dec_conv21_bn(self.dec_conv21(dec13))+enc23)
        dec22 = F.relu(self.dec_conv22_bn(self.dec_conv22(dec21)))
        dec23 = self.up(F.relu(self.dec_conv23_bn(self.dec_conv23(dec22))))
        #stage3
        dec31 = F.relu(self.dec_conv31_bn(self.dec_conv31(dec23))+enc13)
        dec32 = F.relu(self.dec_conv32_bn(self.dec_conv32(dec31)))
        dec33 = self.up(F.relu(self.dec_conv33_bn(self.dec_conv33(dec32))))  
        #final
        fin = F.tanh(self.fin_conv0(dec33))+x
       
        

        return fin



