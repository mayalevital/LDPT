# re-implementation of the net in the paper: Detection of Paroxysmal Atrial Fibrillation using Attention-based Bidirectional Recurrent Neural Networks

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import time

class LastLayers(nn.Module):
    def __init__(self, input_size=50):
        super().__init__()
        hidden_state_size = 50
        self.rnn = nn.RNN(input_size, hidden_state_size, bidirectional=True, batch_first=True)
        
        self.hidden_state = None

        # the rnn output dimension is 100
        self.fc2 = nn.Linear(2*hidden_state_size, 1, bias=False) # for attention layer
        
        self.fc3 = nn.Linear(100, 2)

        self.train_mode = True

    def forward(self, input: Tensor):
        if self.train_mode:
          self.hidden_state = None

        x = input

        # rnn input shape is (batch, seq_len, input_size) when batch_first=True
        x, h = self.rnn(x, self.hidden_state)

        self.hidden_state = h

        # from nn.RNN docs: For the unpacked case, the directions can be separated using output.view(seq_len, batch, num_directions, hidden_size), with forward and backward being direction 0 and 1 respectively. Similarly, the directions can be separated in the packed case.

        # here x shape is (N, seq_len, 100) when batch_first=True

        # attention layer: combine sequences by weighted average. input to attention layer is of size (N, seq_len, 100), out size (N, 100)
        a = self.fc2(x) # shape is (N, seq_len, 1)
        a = nn.Softmax(dim=1)(a)

        x = x.transpose(1, 2) # shape (N, 100, seq_len)


        #multiplying a (N, 100, seq_len) by (N, seq_len, 1) in order to get (N, 100)

        h_att = torch.bmm(x, a) # shape (N, 100, 1)
        h_att = h_att.squeeze(dim=2) # shape (N, 100)
        # this corresponds to h_att in the paper: the output of attention layer

        x = h_att

        layer_output = self.fc3(x)

        # layer_output shape is (N, 2)
        #print(layer_output.shape) # [N, seq_len, 2] when not using attention layer

        return layer_output

    def freeze_all_layers(self):
       for p in self.parameters():
          p.requires_grad = False

    def unfreeze_all_layers(self):
       for p in self.parameters():
          p.requires_grad = True


class MainNet(nn.Module):
    def __init__(self, sigmoid_in_cnn=False):
        super().__init__()
        n_conv_features = 10
        cnn_layers = []
        # the input to first conv layer has 1 feature and size of 20*300.
      
        # (20, 300)->(18, 280)->(16, 260)->(8, 130)->(5,110)->(2,90) - paper input size

        if sigmoid_in_cnn:
            non_linear = nn.Sigmoid()
        else:
            non_linear = nn.ReLU()

        cnn_layers.append(nn.Conv2d(1, n_conv_features, (3, 21), padding=0))
        cnn_layers.append(non_linear)
        cnn_layers.append(nn.Conv2d(n_conv_features, n_conv_features, (3, 21), padding=0))
        cnn_layers.append(non_linear)
        cnn_layers.append(nn.Dropout(0.2))

        cnn_layers.append(nn.MaxPool2d(2))
        cnn_layers.append(nn.Conv2d(n_conv_features, n_conv_features, (4, 21), padding=0))
        cnn_layers.append(non_linear)
        cnn_layers.append(nn.Dropout(0.2))

        cnn_layers.append(nn.Conv2d(n_conv_features, n_conv_features, (4, 21), padding=0))
        cnn_layers.append(non_linear)
        cnn_layers.append(nn.Dropout(0.2))

        self.cnns = nn.Sequential(*cnn_layers)
        self.fc1 = nn.Linear(n_conv_features*2*90, 50)

        # the output of CNN layers: 50 dimension feature vector

        self.last_layers = LastLayers(50)
        self.epochs_performed = 0

    def forward(self, input: Tensor):
        """
        :param input: Batch of sequences of wavelet power spectrum for 30 seconds (non-overlaping windows). input size: (N, seq_len, 20, 300).
        """
        #print(input.shape) #(4, 20, 20, 300)

        N = input.shape[0] # batch size
        seq_len = input.shape[1]
        x = self.cnns(input.view(N*seq_len, 1, 20, 300)) # 1 channel, 20*300 image

        #print(x.shape) # (80, 10, 11, 90)
        x = self.fc1(x.view(N*seq_len, -1))
        x = x.view(N, seq_len, 50) # shape is (N, seq_len, 50)

        layer_output = self.last_layers(x)

        return layer_output

    def probs(self, input: Tensor):
        y_pred_scores = self.forward(input)
        result = torch.nn.Softmax(dim=-1)(y_pred_scores)
        return result

class IntervalNet(nn.Module):
    def __init__(self, input_size=30, last_layers=None):
        super().__init__()
        hidden_state_size = input_size*2
        self.lstm = nn.LSTM(input_size, hidden_state_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_state_size, 50)
        self.drop = nn.Dropout(0.2)
        if not last_layers:
          last_layers = LastLayers(50)
        self.last_layers = last_layers
        self.epochs_performed = 0

    def forward(self, input: Tensor):
        """
        :param input: Batch of sequences of RR interval list. Each list length is 100.
        """
        #print('input shape:', input.shape) # [N, 100, 100]
        #print('input stride:', input.stride())
        N = input.shape[0] # batch size
        seq_len = input.shape[1]
        x, state = self.lstm(input)
        #print('shape after lstn:', x.shape) # [N, seq_len, 2*hidden_state_size]
        #print('stride:', x.stride()) #(400, 1600, 1)
        x = self.fc(x)
        x = self.drop(x)
        #print('shape:', x.shape) # [N, seq_len, 50]
        layer_output = self.last_layers(x)
        return layer_output

    def probs(self, input: Tensor):
        y_pred_scores = self.forward(input)
        result = torch.nn.Softmax(dim=-1)(y_pred_scores)
        return result

    def freeze_all_layers(self):
       for p in self.parameters():
          p.requires_grad = False

    def freeze_last_layers(self):
       for p in self.last_layers.parameters():
          p.requires_grad = False

    def unfreeze_last_layers(self):
       for p in self.last_layers.parameters():
          p.requires_grad = True


def test_train():
	model = MainNet()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	#loss_fn = nn.NLLLoss()
	loss_fn = nn.CrossEntropyLoss()
	#train(model, optimizer, loss_fn, ???)
	X = torch.randn(88, 7, 20, 300)
	y = torch.randint(2, (88, 7))
	
	# Forward pass
	y_pred_log_proba = model(X)

	print(X.shape)
	print(y_pred_log_proba.shape)
	print(y.shape)

	optimizer.zero_grad()
	loss = loss_fn(y_pred_log_proba.reshape(-1, 2), y.view(-1))
	loss.backward()
	optimizer.step()


# this class is taken from tutorial5 (here for reference only):
class RNNLayer(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, phi_h=torch.tanh, phi_y=torch.sigmoid):
        super().__init__()
        self.phi_h, self.phi_y = phi_h, phi_y
        
        self.fc_xh = nn.Linear(in_dim, h_dim, bias=False)
        self.fc_hh = nn.Linear(h_dim, h_dim, bias=True)
        self.fc_hy = nn.Linear(h_dim, out_dim, bias=True)
        
    def forward(self, xt, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(xt.shape[0], self.fc_hh.in_features)
        
        ht = self.phi_h(self.fc_xh(xt) + self.fc_hh(h_prev))
        
        yt = self.fc_hy(ht)
        
        if self.phi_y is not None:
            yt = self.phi_y(yt)
        
        return yt, ht
