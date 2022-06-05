import torch

from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, ct_ht):
        output = torch.tanh(self.l1(ct_ht))
        output = self.dropout(output)
        h_tilda = output
        return h_tilda