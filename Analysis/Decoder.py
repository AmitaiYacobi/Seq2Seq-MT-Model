import torch

from torch import nn
from torch.nn import functional as F

from Attention import *


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, encoder_hidden_dim=128, hidden_dim=128, p=0.1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=p)
        # this vector is mentioned in the paper - https://arxiv.org/pdf/1508.04025.pdf as the third option
        # for the score(ht, hs_hat) value.
        # according to the paper it is suppose to be multiplied by tanh(Wa[ht;hs_hat]). 
        # It gave me much better results!! 
        self.va = nn.Linear(self.hidden_dim,1)
        
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim = self.embedding_dim
        )
        
        self.lstm = nn.LSTM(
            self.embedding_dim, 
            self.hidden_dim
        )
        
        self.score = nn.Linear(
            self.encoder_hidden_dim + self.hidden_dim,
            self.hidden_dim
        )

        self.attention = Attention(
            self.encoder_hidden_dim + self.hidden_dim,
            self.hidden_dim
        )

        self.output_for_softmax = nn.Linear(
            self.hidden_dim,
            self.vocab_size
        )

    def forward(self, symbol, prev_decoder_hidden, encoder_states):
        scores = torch.zeros(len(encoder_states))
        embedded_input = self.embedding(symbol).view(1,1,-1)
        embedded_input = self.dropout(embedded_input)
        ht, next_hidden = self.lstm(embedded_input, prev_decoder_hidden)
        for i in range(len(encoder_states)):
            hs_hat = encoder_states[i]
            # ht;hs_hat 
            ht_hs_hat = torch.cat((ht[0], hs_hat.view(1, -1)), dim=1)
            # tanh(Wa[ht;hs_hat])
            score = torch.tanh(self.score(ht_hs_hat)).view(-1)
            # score(ht,hs_hat) = va * tanh(Wa[ht;hs_hat]) as mentioned in the paper
            scores[i] = self.va(score)
        
        # at(s) = align(ht, hs_hat) - normalize attention scores
        normalized_scores = F.softmax(scores, dim=0)
        normalized_scores = normalized_scores.unsqueeze(0).unsqueeze(0)
        # ct = ∑ (aj*hj_hat)
        ct = normalized_scores.bmm(encoder_states.unsqueeze(0))
        ct_ht = torch.cat((ct, ht), dim=2)
        ht_tilda = torch.tanh(self.attention(ct_ht))
        output = F.log_softmax(self.output_for_softmax(ht_tilda.view(1, -1)), dim=1)
        
        return output, next_hidden, normalized_scores[0]