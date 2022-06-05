
   
import torch
import sacrebleu
import numpy as np

from torch import optim
from torch.nn import functional as F

from Encoder import *
from Decoder import *

from utils import BEGIN_SYMBOL, END_SYMBOL, dump_attention_weights, visualize_attention_weights

class Seq2SeqModel:
    def __init__(
        self,
        train_src_indices,
        train_trg_indices,
        dev_src_indices,
        dev_trg_indices,
        src_vocab,
        trg_vocab,
        dev_src_data,
        dev_trg_data,
        index_to_symbol_trg,
        example_to_plot,
        epochs=10
    ):
        
        self.epochs = epochs
        self.dev_trg_data = dev_trg_data
        self.dev_src_data = dev_src_data
        self.example_to_plot = example_to_plot
        self.index_to_symbol_trg = index_to_symbol_trg

        # Make the refs to be one list of sequences from the dev target data (not list of lists)
        # because that is how the refs should be represented for the corpus_bleu function
        self.refs = [' '.join(ref) for ref in self.dev_trg_data]

        self.train_src_input = [torch.LongTensor(seq) for seq in train_src_indices]
        self.train_trg_input = [torch.LongTensor(seq) for seq in train_trg_indices]
        
        self.dev_src_input = [torch.LongTensor(seq) for seq in dev_src_indices]
        self.dev_trg_input = [torch.LongTensor(seq) for seq in dev_trg_indices]
        

        self.encoder = Encoder(vocab_size=len(src_vocab), embedding_dim=128, hidden_dim=256)
        self.decoder = Decoder(vocab_size=len(trg_vocab), embedding_dim=128, encoder_hidden_dim=256, hidden_dim=256)

        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=0.0005)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=0.0005)
        
        self.loss_function = nn.NLLLoss()
        
    def epoch(self, src, trg):
        epoch_loss = 0
        for i , (src_seq, trg_seq) in enumerate(zip(src, trg)):
            seq_loss = 0
            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()
            
            src_seq_len = src_seq.size(0)
            trg_seq_len = trg_seq.size(0)
            
            encoder_hidden_state = torch.zeros(1, 1, self.encoder.hidden_dim), torch.zeros(1, 1, self.encoder.hidden_dim)
            decoder_hidden_state = torch.zeros(1, 1, self.decoder.hidden_dim), torch.zeros(1, 1, self.decoder.hidden_dim)
            encoder_states = torch.zeros(src_seq_len, self.encoder.hidden_dim)

            for i in range(src_seq_len):
                encoder_output, encoder_hidden_state = self.encoder(src_seq[i], encoder_hidden_state)
                encoder_states[i] = encoder_output.view(-1)

            for i in range(trg_seq_len - 1):
                decoder_output, decoder_hidden_state, _ = self.decoder(trg_seq[i], decoder_hidden_state, encoder_states)
                next_symbol = trg_seq[i + 1].unsqueeze(0)

                # Sum the loss of every symbol in the current sequence
                seq_loss += self.loss_function(decoder_output, next_symbol)
            
            epoch_loss += (seq_loss.item())
            seq_loss.backward()
            
            self.encoder_optim.step()
            self.decoder_optim.step()
            
        return epoch_loss
    
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
            
            # Make the core of the epoch
            epoch_loss = self.epoch(src, trg)
            
            # Validation on dev set
            dev_loss, bleu_score = self.validate()
            
            # Analysis
            self.analyze(epoch)

            # Average loss for the current epoch
            train_loss = epoch_loss / len(train_src)

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
        epoch_loss = 0
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
                encoder_states = torch.zeros(src_seq_len, self.encoder.hidden_dim)


                for i in range(src_seq_len):
                    encoder_output, encoder_hidden_state = self.encoder(src_seq[i], encoder_hidden_state)
                    encoder_states[i] = encoder_output.view(-1)


                for i in range(trg_seq_len - 1):
                    decoder_output, decoder_hidden_state, _ = self.decoder(trg_seq[i], decoder_hidden_state, encoder_states)
                    next_symbol = trg_seq[i + 1].unsqueeze(0)

                    # Sum the loss of every symbol in the current sequence
                    seq_loss += self.loss_function(decoder_output, next_symbol)
                
                epoch_loss += (seq_loss.item())
                output_symbols = self.predict_output_symbols(trg_seq[0], encoder_states)
                predictions.append(' '.join(output_symbols))

            dev_loss = epoch_loss / len(dev_src)
            bleu = sacrebleu.corpus_bleu(predictions, [self.refs])
        
        return dev_loss, bleu.score

           
    def predict_output_symbols(self, first_symbol, encoder_states):
        output_symbols = []
        decoder_next_input = first_symbol
        max_trg_seq_len = max([len(seq) for seq in self.dev_trg_input])
        decoder_hidden_state = torch.zeros(1, 1, self.decoder.hidden_dim), torch.zeros(1, 1, self.decoder.hidden_dim)

        for i in range(max_trg_seq_len):
            decoder_output, decoder_hidden_state, _ = self.decoder(decoder_next_input, decoder_hidden_state, encoder_states)
            decoder_next_input = np.argmax(decoder_output.data, axis=1)
            predicted_symbol = self.index_to_symbol_trg[decoder_next_input.item()]
            output_symbols.append(predicted_symbol)

            # Stop if the model predicted the end symbol - '</s>'
            if predicted_symbol == END_SYMBOL:
                break
            
        output_symbols = [BEGIN_SYMBOL] + output_symbols
        return output_symbols

    def analyze(self, epoch):
        
        src_example = self.dev_src_input[self.example_to_plot]
        trg_example = self.dev_trg_input[self.example_to_plot]
        raw_src_example = self.dev_src_data[self.example_to_plot]
        raw_trg_example = self.dev_trg_data[self.example_to_plot]
        attention_weights = []

        with torch.no_grad():

            src_seq_len = src_example.size(0)
            trg_seq_len = trg_example.size(0)
            
            encoder_hidden_state = torch.zeros(1, 1, self.encoder.hidden_dim), torch.zeros(1, 1, self.encoder.hidden_dim)
            decoder_hidden_state = torch.zeros(1, 1, self.decoder.hidden_dim), torch.zeros(1, 1, self.decoder.hidden_dim)
            encoder_states = torch.zeros(src_seq_len, self.encoder.hidden_dim)
            decoder_next_input = trg_example[0]

            for i in range(src_seq_len):
                encoder_output, encoder_hidden_state = self.encoder(src_example[i], encoder_hidden_state)
                encoder_states[i] = encoder_output.view(-1)


            for i in range(trg_seq_len - 1):
                decoder_output, decoder_hidden_state, weights = self.decoder(decoder_next_input, decoder_hidden_state, encoder_states)
                decoder_next_input = np.argmax(decoder_output.data, axis=1)
                attention_weights.append(weights.data)
            
            attention_weights = torch.cat(attention_weights, dim=0)

            visualize_attention_weights(
                "Epoch_" + str(epoch), 
                raw_src_example[1:src_seq_len - 1], 
                raw_trg_example[1:trg_seq_len - 1], 
                attention_weights
            )

            dump_attention_weights(attention_weights, epoch)
