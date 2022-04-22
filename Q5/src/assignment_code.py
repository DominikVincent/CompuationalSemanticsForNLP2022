import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import math


# you can refer to the implementation provided by PyTorch for more information
# https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
"""
Parameters
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

Inputs: input, hidden
input of shape (batch, input_size): tensor containing input features
hidden of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.

Outputs: h'
h’ of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
"""
class GRUCell_assignment(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell_assignment, self).__init__()
        # hyper-parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        #############################################################
        #  please initialize the parameters (weight matrices) here  #
        #############################################################
        # *you may use torch.nn.Linear() as weight matrix and bias  #
        #############################################################

        self.linear_wir = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_whr = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_wiz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_whz = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_Win = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_whn = torch.nn.Linear(hidden_size, hidden_size, bias=bias)





    def forward(self, inputs, hidden=False):
        if hidden is False: hidden = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        x, h = inputs, hidden
        ##############################################################
        #  refer to the GRU equations and write down your code here  #
        ##############################################################
        #        *you may use torch.tanh() as the tanh function      #
        #    *you may use torch.sigmoid() as the sigmoid function    #
        #    *you may use A*B as the Hadamard product for A and B    #
        ##############################################################

        r = torch.sigmoid(self.linear_wir(x) + self.linear_whr(h))
        z = torch.sigmoid(self.linear_wiz(x) + self.linear_whz(h))

        n = torch.tanh(self.linear_Win(x) + r * self.linear_whn(h))

        h = (1 - z) * n + z * h
        return h


# you can refer to the implementation provided by PyTorch for more information
# https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
"""
Parameters
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

Inputs: input, (h_0, c_0)
input of shape (batch, input_size): tensor containing input features
h_0 of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
c_0 of shape (batch, hidden_size): tensor containing the initial cell state for each element in the batch.
If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

Outputs: (h_1, c_1)
h_1 of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
c_1 of shape (batch, hidden_size): tensor containing the next cell state for each element in the batch
"""
class LSTMCell_assignment(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell_assignment, self).__init__()
        # hyper-parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        #############################################################
        #  please initialize the parameters (weight matrices) here  #
        #############################################################
        # *you may use torch.nn.Linear() as weight matrix and bias  #
        #############################################################
        self.linear_wii = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_whi = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_wif = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_whf = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_wig = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_whg = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.linear_wio = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_who = torch.nn.Linear(hidden_size, hidden_size, bias=bias)








    def forward(self, inputs, h_0=False, c_0=False):
        if h_0 is False: h_0 = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        if c_0 is False: c_0 = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        x, h, c = inputs, h_0, c_0
        ##############################################################
        #  refer to the LSTM equations and write down your code here #
        ##############################################################
        #        *you may use torch.tanh() as the tanh function      #
        #    *you may use torch.sigmoid() as the sigmoid function    #
        #    *you may use A*B as the Hadamard product for A and B    #
        ##############################################################

        i = torch.sigmoid(self.linear_wii(x) + self.linear_whi(h))
        f = torch.sigmoid(self.linear_wif(x) + self.linear_whf(h))
        o = torch.sigmoid(self.linear_wio(x) + self.linear_who(h))
        g = torch.tanh(self.linear_wig(x) + self.linear_whg(h))

        c = f * c + i * g
        h = o * torch.tanh(c)


        return h, c