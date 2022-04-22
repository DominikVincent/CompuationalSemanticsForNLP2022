import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import math
from assignment_code import GRUCell_assignment

class GRUModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = GRUCell_assignment(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        #text = [sent len, batch size]
        embedded = self.embedding(text)
        #embedded = [sent len, batch size, emb dim]
        hidden = embedded.new_zeros(embedded.shape[1], self.hidden_dim)
        for seq in range(embedded.shape[0]):
            input_seq = embedded[seq,:,:]
            hidden = self.gru(input_seq, hidden)

        return self.fc(hidden.squeeze(0))
