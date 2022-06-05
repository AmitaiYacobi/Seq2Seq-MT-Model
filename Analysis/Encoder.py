import torch

from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, p=0.1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=p)
        
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=self.embedding_dim
        )
        
        self.lstm = nn.LSTM(
            self.embedding_dim, 
            self.hidden_dim
        )

    def forward(self, symbol, hidden_state):
        embedded_input = self.embedding(symbol).view(1,1,-1)
        embedded_input = self.dropout(embedded_input)
        output, next_hidden = self.lstm(embedded_input, hidden_state)
        return output, next_hidden