import sys
import time
from cv2 import selectROI
import torch
import sacrebleu
import numpy as np

from torch import nn, tensor
from torch import optim
from torch.nn import functional as F

from utils import *

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, p=0.3):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.dropout = nn.Dropout(p=p)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim = self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)

    def forward(self, symbol, hidden_state):
        embedded_input = self.embedding(symbol).view(1,1,-1)
        # print(embedded_input.size())
        # print(hidden_state[0].size())
        # return
        embedded_input = self.dropout(embedded_input)
        output, next_hidden = self.lstm(embedded_input, hidden_state)
        return output, next_hidden

    def initial_hidden_state(self):
        hidden_state = torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim)
        return hidden_state

class Decoder(nn.Module):
    def __init__(self, vocab_size, encoder_out_dim=128, embedding_dim=128, hidden_dim=256, p=0.3):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_out_dim = encoder_out_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.dropout = nn.Dropout(p=p)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim = self.embedding_dim)
        
        # Here we concatenating the encoder output and the embedding of the decoder input sequence
        self.lstm = nn.LSTM(self.embedding_dim + self.encoder_out_dim, self.hidden_dim)
        self.output_for_softmax = nn.Linear(self.hidden_dim, self.vocab_size)
        

    def forward(self, symbol, hidden_state, encoder_output):
        embedded_input = self.embedding(symbol).view(1,1,-1)
        embedded_input = self.dropout(embedded_input)

        concatenated_input = torch.cat([embedded_input, encoder_output], dim=2)
        output, next_hidden = self.lstm(concatenated_input, hidden_state)
        output = F.log_softmax(self.output_for_softmax(output.view(1,-1)), dim=1)
        
        return output, next_hidden
    
    def initial_hidden_state(self):
        hidden_state = torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim)
        return hidden_state
        
class Seq2SeqModel:
    def __init__(
        self,
        train_src_tensors,
        train_trg_tensors,
        dev_src_tensors,
        dev_trg_tensors,
        dev_trg_data,
        encoder,
        decoder,
        encoder_optim,
        decoder_optim,
        loss_function,
        index_to_symbol_trg, 
        epochs=15
    ):
        
        self.epochs = epochs
        self.train_src_tensors = train_src_tensors
        self.train_trg_tensors = train_trg_tensors
        self.dev_src_tensors = dev_src_tensors
        self.dev_trg_tensors = dev_trg_tensors
        self.dev_trg_data = dev_trg_data
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optim = encoder_optim
        self.decoder_optim = decoder_optim
        self.loss_function = loss_function
        self.index_to_symbol_trg = index_to_symbol_trg
        
        
    def epoch(self, src, trg):
        loss_sum = 0
        for i , (src_seq, trg_seq) in enumerate(zip(src, trg)):
            seq_loss = 0
            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()
            
            src_seq_len = src_seq.size(0)
            trg_seq_len = trg_seq.size(0)
            
            encoder_hidden_state = self.encoder.initial_hidden_state()
            decoder_hidden_state = self.decoder.initial_hidden_state()
            
            for i in range(src_seq_len - 1):
                encoder_output, encoder_hidden_state = self.encoder(src_seq[i], encoder_hidden_state)
                # encoder_output = self.encoder(src_seq)
            
            for i in range(trg_seq_len - 1):
                decoder_output, decoder_hidden_state = self.decoder(trg_seq[i], decoder_hidden_state, encoder_output)
                next_symbol = trg_seq[i + 1].unsqueeze(0)
                # Sum the loss of every symbol in the current sequence
                seq_loss += self.loss_function(decoder_output, next_symbol)
            
            loss_sum += (seq_loss.item() / (trg_seq_len - 1))
            seq_loss.backward()
            
            self.encoder_optim.step()
            self.decoder_optim.step()
            
            return loss_sum
    
    def shuffle_data(self):
        zipped_data = list(zip(self.train_src_tensors, self.train_trg_tensors))
        np.random.shuffle(zipped_data)
        return zip(*zipped_data)
          
    def train(self):
        dev_losses = []
        bleu_scores = []
        best_score = 0
        for epoch in range(1, self.epochs + 1):
            self.encoder.train()
            self.decoder.train()
            
            # Shuffle the data in every epoch
            train_src, train_trg = self.shuffle_data()
            loss = self.epoch(train_src, train_trg)
            # Average loss for the current epoch
            train_loss = loss / len(train_src)
            dev_loss, bleu =  self.test()

            dev_losses.append(dev_loss)
            bleu_scores.append(bleu.score)

            if bleu.score > best_score:
                best_score = bleu.score
                torch.save(self.encoder.state_dict(), "./results/encoder")
                torch.save(self.decoder.state_dict(), "./results/decoder")

            print(f"Epoch: {epoch}/{self.epochs} | " +
                  f"Train Loss: {'{:.4f}'.format(train_loss)} | " +
                  f"Dev Loss: {'{:.4f}'.format(dev_loss)} | " + 
                  f"Bleu score: {'{:.4f}'.format(bleu.score)}")
        
        print("\n###############################")
        print("Finished training the model!")
        print("###############################")
        return dev_losses, bleu_scores
        
    def test(self):
        loss_sum = 0
        # Here we will store all predicted sequences from the model
        predictions = []
        
        dev_src = self.dev_src_tensors
        dev_trg = self.dev_trg_tensors
        
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            for i ,(src_seq, trg_seq) in enumerate(zip(dev_src, dev_trg)):
                seq_loss = 0

                src_seq_len = src_seq.size(0)
                trg_seq_len = trg_seq.size(0)
                
                encoder_hidden_state = self.encoder.initial_hidden_state()
                decoder_hidden_state = self.decoder.initial_hidden_state()
                
                for i in range(src_seq_len - 1):
                    encoder_output, encoder_hidden_state = self.encoder(src_seq[i], encoder_hidden_state)
                
                for i in range(trg_seq_len - 1):
                    decoder_output, decoder_hidden_state = self.decoder(trg_seq[i], decoder_hidden_state, encoder_output)
                    next_symbol = trg_seq[i + 1].unsqueeze(0)
                    # Sum the loss of every symbol in the current sequence
                    seq_loss += self.loss_function(decoder_output, next_symbol)
                
                loss_sum += (seq_loss.item() / (trg_seq_len - 1))
                output_symbols, begin_symbol = self.predict_output_symbols(trg_seq[0], encoder_output)
                predictions.append(' '.join([begin_symbol] + output_symbols))
            dev_loss = loss_sum / len(dev_src)
            refs = [' '.join(ref) for ref in self.dev_trg_data]
            # return
            bleu = sacrebleu.corpus_bleu(predictions, [refs])
        
        return dev_loss, bleu
           
    def predict_output_symbols(self, first_symbol, encoder_output, begin_symbol='<s>', end_symbol='</s>'):
        max_trg_seq_len = max([len(seq) for seq in self.dev_trg_tensors])
        hidden_state = self.decoder.initial_hidden_state()
        
        output_symbols = []
        decoder_next_input = first_symbol

        for i in range(max_trg_seq_len):
            decoder_output, hidden_state = self.decoder(decoder_next_input, hidden_state, encoder_output)
            decoder_next_input = np.argmax(decoder_output.data, axis=1)
            predicted_symbol = self.index_to_symbol_trg[decoder_next_input.item()]
            output_symbols.append(predicted_symbol)
            # print(self.index_to_symbol_trg)
            # Stop if the model predicted the end symbol - '</s>'
            if predicted_symbol == end_symbol:
                break
            
        return output_symbols, begin_symbol
   
    
def main():
    train_src_file = sys.argv[1]
    train_trg_file = sys.argv[2]
    dev_src_file = sys.argv[3]
    dev_trg_file = sys.argv[4]
    
    train_src_indices, train_trg_indices, src_vocab, trg_vocab, _ = prepare_data(train_src_file, train_trg_file)
    dev_src_indices, dev_trg_indices, _, _, dev_trg_data = prepare_data(dev_src_file, dev_trg_file)
    
    index_to_symbol_trg = indices_to_symbols(trg_vocab)
    # print(train_src_indices[0])
    
    train_src_tensors = [torch.LongTensor(seq) for seq in train_src_indices]
    train_trg_tensors = [torch.LongTensor(seq) for seq in train_trg_indices]
    
    dev_src_tensors = [torch.LongTensor(seq) for seq in dev_src_indices]
    dev_trg_tensors = [torch.LongTensor(seq) for seq in dev_trg_indices]
    
    encoder = Encoder(vocab_size=len(src_vocab), embedding_dim=128, hidden_dim=128)
    decoder = Decoder(vocab_size= len(trg_vocab), embedding_dim=128, hidden_dim=256)
    
    encoder_optim = optim.Adam(encoder.parameters(), lr=0.0005)
    decoder_optim = optim.Adam(decoder.parameters(), lr=0.0005)
    
    loss_function = nn.NLLLoss()
    
    create_results_dir()
    model = Seq2SeqModel(
        train_src_tensors,
        train_trg_tensors,
        dev_src_tensors,
        dev_trg_tensors,
        dev_trg_data,
        encoder,
        decoder,
        encoder_optim,
        decoder_optim,
        loss_function,
        index_to_symbol_trg,
        epochs=40
    )
    
    losses, bleu_scores = model.train()
    


if __name__ == "__main__":
    main()
