import numpy as np
import random
import wfdb
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
import function
import main_sig_process

# 10 minutes segments with lables
class SegmentDataset(Dataset):
    def __init__(self, patient, device='cuda', seq_len=20, rr=False):
        self.device = device
        self.seq_len = seq_len
        if rr:
          data = torch.load(patient + '/rr_intervals.dat')  # list of list of numpy.int64
          labels = torch.load(patient + '/rr_label.dat') # list of numpy.int64
        else:
          #wavelets = torch.load(patient + '/wavelets.dat')  # list of ndarrays
          data = torch.load(patient + '/wavelets.dat')  # list of tensors of type float64
          labels = torch.load(patient + '/labels.dat') # list of ints
          data = torch.stack(data)
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        self.n = data.shape[0]
        labels = torch.as_tensor(labels, device=self.device)
        self.data = data
        self.labels = labels
    # return a tensor of shape (seq_len, W, H)
    def __getitem__(self, index):
      n = self.seq_len
      label = self.labels[index] # 1 label per segment
      #labels = label * torch.ones(self.seq_len, dtype=torch.long, device=self.device) # duplicate 1 label for 20 windows
      #labels = self.labels[index*n:(index+1)*n] # 1 label per 30 sec
      return self.data[index*n:(index+1)*n], label
    def __len__(self):
        return self.n // self.seq_len


class PatientDataset(Dataset):
    def __init__(self, folder, channels):
        self.sig, fields = wfdb.rdsamp(folder, channels=channels)
        self.fs = fields['fs']
        self.n = 30 * self.fs

        self.folder = folder
        self.channels = channels
        self.n_items = function.samples_in_record(folder) / self.n

        #for debugging:
        #print('self.n:', self.n)
        #print('self.fs:', self.fs)

    # each item is an ndarray of representing a 30 seconds window
    def __getitem__(self, index):
        n = self.n
        sig = self.sig[index*n : (index+1)*n, :]
        return sig

    def __len__(self):
        return self.n_items
