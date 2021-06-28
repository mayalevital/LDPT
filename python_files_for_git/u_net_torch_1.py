#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:05:48 2021

@author: maya.fi@bm.technion.ac.il
"""

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
class Net(nn.Module):
    
 
        
    def __init__(self, params):
        super(Net, self).__init__()
        
        self.batch_size = params['batch_size']
        self.kernel_size = params['ker_size']
        self.multi_slice_n = params['multi_slice_n']
        enc_d = params['encoder_depth']
        cen_d = params['center_depth'] 
        dec_d = params['decoder_depth'] 
        pad = int((self.kernel_size-1)/2)
        self.num_chan = params['num_chan']
        #encoder
        #stage 1
        #if(multi_slice_n == 1):
            #conv = nn.Conv2d
            #batchnorm = nn.BatchNorm2d
            #MaxPool = nn.MaxPool2d(2,2)
             
        #else:
        conv = nn.Conv3d
        batchnorm = nn.BatchNorm3d
        MaxPool = nn.MaxPool3d((2,2,1))
            
        self.enc_conv11 = conv(self.num_chan, enc_d[0], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv11_bn = batchnorm(enc_d[0])
        
        self.enc_conv12 = conv(enc_d[0], enc_d[0], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv12_bn = batchnorm(enc_d[0])
        self.enc_conv13 = conv(enc_d[0], enc_d[0], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv13_bn = batchnorm(enc_d[0])
        #stage 2
        self.enc_conv21 = conv(enc_d[0], enc_d[1], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv21_bn = batchnorm(enc_d[1])
        self.enc_conv22 = conv(enc_d[1], enc_d[1], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv22_bn = batchnorm(enc_d[1])
        self.enc_conv23 = conv(enc_d[1], enc_d[1], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv23_bn = batchnorm(enc_d[1])
        #stage 3
        self.enc_conv31 = conv(enc_d[1], enc_d[2], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv31_bn = batchnorm(enc_d[2])
        self.enc_conv32 = conv(enc_d[2], enc_d[2], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv32_bn = batchnorm(enc_d[2])
        self.enc_conv33 = conv(enc_d[2], enc_d[2], kernel_size=self.kernel_size, padding=pad)
        self.enc_conv33_bn = batchnorm(enc_d[2])  
        #center
        self.cen_conv11 = conv(enc_d[2], cen_d[0], kernel_size=self.kernel_size, padding=pad)
        self.cen_conv11_bn = batchnorm(cen_d[0])
        self.cen_conv12 = conv(cen_d[0], cen_d[0], kernel_size=self.kernel_size, padding=pad)
        self.cen_conv12_bn = batchnorm(cen_d[0])
        self.cen_conv13 = conv(cen_d[0], cen_d[0], kernel_size=self.kernel_size, padding=pad)
        self.cen_conv13_bn = batchnorm(cen_d[0])
        self.cen_conv13_sm = conv(cen_d[0], cen_d[0], kernel_size=self.kernel_size, padding=pad)

        #decoder
        #stage 1
        self.dec_conv11 = conv(enc_d[2]+cen_d[0], dec_d[0], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv11_bn = batchnorm(dec_d[0])
        self.dec_conv12 = conv(dec_d[0], dec_d[0], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv12_bn = batchnorm(dec_d[0])
        self.dec_conv13 = conv(dec_d[0], dec_d[0], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv13_bn = batchnorm(dec_d[0])
        self.dec_conv13_sm = conv(dec_d[0], dec_d[0], kernel_size=self.kernel_size, padding=pad)
        #stage 2
        self.dec_conv21 = conv(enc_d[1]+dec_d[0], dec_d[1], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv21_bn = batchnorm(dec_d[1])
        self.dec_conv22 = conv(dec_d[1], dec_d[1], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv22_bn = batchnorm(dec_d[1])
        self.dec_conv23 = conv(dec_d[1], dec_d[1], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv23_bn = batchnorm(dec_d[1])
        self.dec_conv23_sm = conv(dec_d[1], dec_d[1], kernel_size=self.kernel_size, padding=pad)

        #stage 3
        self.dec_conv31 = conv(enc_d[0]+dec_d[1], dec_d[2], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv31_bn = batchnorm(dec_d[2])
        self.dec_conv32 = conv(dec_d[2], dec_d[2], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv32_bn = batchnorm(dec_d[2])
        self.dec_conv33 = conv(dec_d[2], dec_d[2], kernel_size=self.kernel_size, padding=pad)
        self.dec_conv33_bn = batchnorm(dec_d[2])
        self.dec_conv33_sm = conv(dec_d[2], dec_d[2], kernel_size=self.kernel_size, padding=pad)
        #final
        self.fin_conv0 = conv(dec_d[2], 1, kernel_size=1)
        #other operations
        self.pool = MaxPool

    
    def forward(self, x):
        #stage1
        enc11 = F.relu(self.enc_conv11_bn(self.enc_conv11(x)))
        fin_enc13 = self.pool(enc11)
        #stage 2
        enc21 = F.relu(self.enc_conv21_bn(self.enc_conv21(fin_enc13)))
        fin_enc23 = self.pool(enc21)
        #stage3
        enc31 = F.relu(self.enc_conv31_bn(self.enc_conv31(fin_enc23)))
        fin_enc33 = self.pool(enc31)
        #center
        cen11 = F.relu(self.cen_conv11_bn(self.cen_conv11(fin_enc33)))
        upsamp_size = (cen11.size()[2]*2, cen11.size()[2]*2, self.multi_slice_n)
        upsamp = nn.Upsample(upsamp_size, mode = 'trilinear')
        fin_cen = upsamp(cen11)
        #decoder
        #stage1
  
        in_dec11 = torch.cat((enc31, fin_cen), dim=1)
        dec11 = F.relu(self.dec_conv11_bn(self.dec_conv11(in_dec11)))
        upsamp_size = (dec11.size()[2]*2, dec11.size()[2]*2, self.multi_slice_n)
        upsamp = nn.Upsample(upsamp_size, mode = 'trilinear')
        fin_dec13 = upsamp(dec11)
        #stage 2
        in_dec21 = torch.cat((enc21, fin_dec13), dim=1)
        dec21 = F.relu(self.dec_conv21_bn(self.dec_conv21(in_dec21)))
        upsamp_size = (dec21.size()[2] * 2, dec21.size()[2] * 2, self.multi_slice_n)
        upsamp = nn.Upsample(upsamp_size, mode = 'trilinear')
        fin_dec23 = upsamp(dec21)
        #stage3
        in_dec31 = torch.cat((enc11, fin_dec23), dim=1)
        dec31 = F.relu(self.dec_conv31_bn(self.dec_conv31(in_dec31)))

        #final
        #no tanh
        #reduce the number of convs
        fin = F.sigmoid(self.fin_conv0(dec31))+x
        
        return fin



