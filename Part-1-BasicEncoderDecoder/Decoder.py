import torch

from torch import nn
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(self, vocab_size, encoder_out_dim=128, embedding_dim=128, hidden_dim=128, p=0.1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_out_dim = encoder_out_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=p)
        
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim = self.embedding_dim
            )
        
        # Here we concatenating the encoder output and the embedding of the decoder input sequence
        self.lstm = nn.LSTM(
            self.embedding_dim + self.encoder_out_dim, 
            self.hidden_dim
            )

        self.output_for_softmax = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, symbol, hidden_state, encoder_output):
        embedded_input = self.embedding(symbol).view(1,1,-1)
        embedded_input = self.dropout(embedded_input)

        concatenated_input = torch.cat([embedded_input, encoder_output], dim=2)
        output, next_hidden = self.lstm(concatenated_input, hidden_state)
        output = F.log_softmax(self.output_for_softmax(output.view(1,-1)), dim=1)
        
        return output, next_hidden