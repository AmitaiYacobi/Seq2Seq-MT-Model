import torch
import sacrebleu
import numpy as np

from torch import optim
from torch.nn import functional as F

from Encoder import *
from Decoder import *

class Seq2SeqModel:
    def __init__(
        self,
        train_src_indices,
        train_trg_indices,
        dev_src_indices,
        dev_trg_indices,
        src_vocab,
        trg_vocab,
        dev_trg_data,
        index_to_symbol_trg,
        epochs=10
    ):
        
        self.epochs = epochs
        self.dev_trg_data = dev_trg_data
        self.index_to_symbol_trg = index_to_symbol_trg

        self.train_src_input = [torch.LongTensor(seq) for seq in train_src_indices]
        self.train_trg_input = [torch.LongTensor(seq) for seq in train_trg_indices]
        
        self.dev_src_input = [torch.LongTensor(seq) for seq in dev_src_indices]
        self.dev_trg_input = [torch.LongTensor(seq) for seq in dev_trg_indices]
        
        self.encoder = Encoder(vocab_size=len(src_vocab), embedding_dim=128, hidden_dim=128)
        self.decoder = Decoder(vocab_size= len(trg_vocab), embedding_dim=128, hidden_dim=256)
        
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=0.0007)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=0.0007)
        
        self.loss_function = nn.NLLLoss()
        
        
    def epoch(self, src, trg):
        loss_sum = 0
        for i , (src_seq, trg_seq) in enumerate(zip(src, trg)):
            seq_loss = 0
            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()
            
            src_seq_len = src_seq.size(0)
            trg_seq_len = trg_seq.size(0)
            
            encoder_hidden_state = torch.zeros(1, 1, self.encoder.hidden_dim), torch.zeros(1, 1, self.encoder.hidden_dim)
            decoder_hidden_state = torch.zeros(1, 1, self.decoder.hidden_dim), torch.zeros(1, 1, self.decoder.hidden_dim)
            
            for i in range(src_seq_len):
                encoder_output, encoder_hidden_state = self.encoder(src_seq[i], encoder_hidden_state)
            
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
    
    def shuffle_data(self, src, trg):
        zipped_data = list(zip(src, trg))
        np.random.shuffle(zipped_data)
        return zip(*zipped_data)
          
    def train(self):
        dev_losses = []
        train_losses = []
        bleu_scores = []
        best_score = 0

        train_src = self.train_src_input
        train_trg = self.train_trg_input

        for epoch in range(self.epochs):
            
            self.encoder.train()
            self.decoder.train()

            # Shuffle the data in every epoch
            src, trg = self.shuffle_data(train_src, train_trg)
            
            # Epoch
            epoch_loss = self.epoch(src, trg)

            # Average loss for the current epoch
            train_loss = epoch_loss / len(train_src)
            dev_loss, bleu_score = self.validate()

            train_losses.append(train_loss)
            dev_losses.append(dev_loss)

            if bleu_score > best_score:
                best_score = bleu_score
                bleu_scores.append(bleu_score)
                torch.save(self.encoder.state_dict(), "./results/encoder")
                torch.save(self.decoder.state_dict(), "./results/decoder")
            else:
                bleu_scores.append(best_score)

            print(f"Epoch: {epoch}:  " +
                  f"train Loss: {'{:.4f}'.format(train_loss)} | " +
                  f"dev Loss: {'{:.4f}'.format(dev_loss)} | " + 
                  f"Bleu score: {'{:.4f}'.format(best_score)}")
        
        print("\n###############################")
        print("Finished training the model!")
        print("###############################")
        return train_losses, dev_losses, bleu_scores
        
    def validate(self):
        loss_sum = 0
        predictions = []
        
        dev_src = self.dev_src_input
        dev_trg = self.dev_trg_input
        
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            for i ,(src_seq, trg_seq) in enumerate(zip(dev_src, dev_trg)):
                seq_loss = 0

                src_seq_len = src_seq.size(0)
                trg_seq_len = trg_seq.size(0)
                
                encoder_hidden_state = torch.zeros(1, 1, self.encoder.hidden_dim), torch.zeros(1, 1, self.encoder.hidden_dim)
                decoder_hidden_state = torch.zeros(1, 1, self.decoder.hidden_dim), torch.zeros(1, 1, self.decoder.hidden_dim)
                
                for i in range(src_seq_len):
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
            bleu = sacrebleu.corpus_bleu(predictions, [refs])
        
        return dev_loss, bleu.score

           
    def predict_output_symbols(self, first_symbol, encoder_output, begin_symbol='<s>', end_symbol='</s>'):
        max_trg_seq_len = max([len(seq) for seq in self.dev_trg_input])
        hidden_state = torch.zeros(1, 1, self.decoder.hidden_dim), torch.zeros(1, 1, self.decoder.hidden_dim)
        
        output_symbols = []
        decoder_next_input = first_symbol

        for i in range(max_trg_seq_len):
            decoder_output, hidden_state = self.decoder(decoder_next_input, hidden_state, encoder_output)
            decoder_next_input = np.argmax(decoder_output.data, axis=1)
            predicted_symbol = self.index_to_symbol_trg[decoder_next_input.item()]
            output_symbols.append(predicted_symbol)

            # Stop if the model predicted the end symbol - '</s>'
            if predicted_symbol == end_symbol:
                break
            
        return output_symbols, begin_symbol