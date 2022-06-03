import sys
import torch
import optparse
import sacrebleu
import numpy as np
import matplotlib.pyplot as plt
 
from Encoder import *
from Decoder import *
from utils import *

def predict_output_symbols(decoder, index_to_symbol_trg, first_symbol, encoder_output, trg_input):

    output_symbols = []
    max_trg_seq_len = max([len(seq) for seq in trg_input])
    
    decoder_hidden_state = torch.zeros(1, 1, decoder.hidden_dim), torch.zeros(1, 1, decoder.hidden_dim)
    decoder_next_input = first_symbol

    for i in range(max_trg_seq_len):
        decoder_output, decoder_hidden_state = decoder(decoder_next_input, decoder_hidden_state, encoder_output)
        decoder_next_input = np.argmax(decoder_output.data, axis=1)
        predicted_symbol = index_to_symbol_trg[decoder_next_input.item()]
        output_symbols.append(predicted_symbol)

        # Stop if the model predicted the end symbol - '</s>'
        if predicted_symbol == END_SYMBOL:
            break

    output_symbols = [BEGIN_SYMBOL] + output_symbols    
    return output_symbols

def test(src_input, trg_input, encoder, decoder, index_to_symbol_trg, test_trg_data):

    predictions = []
    refs = [' '.join(ref) for ref in test_trg_data]
    
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for i ,(src_seq, trg_seq) in enumerate(zip(src_input, trg_input)):

            src_seq_len = src_seq.size(0)
            encoder_hidden_state = torch.zeros(1, 1, encoder.hidden_dim), torch.zeros(1, 1, encoder.hidden_dim)

            for i in range(src_seq_len):
                encoder_output, encoder_hidden_state = encoder(src_seq[i], encoder_hidden_state)
        
            output_symbols = predict_output_symbols(decoder, index_to_symbol_trg, trg_seq[0], encoder_output, trg_input)
            predictions.append(' '.join(output_symbols))

        bleu = sacrebleu.corpus_bleu(predictions, [refs])
    
    return  bleu.score

def main():
    parser = optparse.OptionParser()
    parser.add_option("-b", dest="is_should_run_best_results", default="no")
    
    (opts, args) = parser.parse_args()

    test_src_file = sys.argv[1]
    test_trg_file = sys.argv[2]
  

    test_src, test_trg = read_data(test_src_file, test_trg_file)
    
    vocabs = torch.load("./vocabs/vocabs")
    src_vocab = vocabs['src_vocab']
    trg_vocab = vocabs['trg_vocab']
    encoder = Encoder(vocab_size=len(src_vocab), embedding_dim=128, hidden_dim=256)
    decoder = Decoder(vocab_size=len(trg_vocab), encoder_out_dim=256, embedding_dim=128, hidden_dim=256)
    
    if opts.is_should_run_best_results == "yes":
        encoder.load_state_dict(torch.load('./best_results/encoder')) 
        decoder.load_state_dict(torch.load('./best_results/decoder'))
    else:
        encoder.load_state_dict(torch.load('./results/encoder')) 
        decoder.load_state_dict(torch.load('./results/decoder'))
    
    symbol_to_index_src, index_to_symbol_src = map_symbols_and_indices(src_vocab)
    symbol_to_index_trg, index_to_symbol_trg = map_symbols_and_indices(trg_vocab)

    test_src_indices = symbols_to_indices(test_src, symbol_to_index_src)
    test_trg_indices = symbols_to_indices(test_trg, symbol_to_index_trg) 

    src_input = [torch.LongTensor(x) for x in test_src_indices]
    trg_input = [torch.LongTensor(x) for x in test_trg_indices]

    bleu_score = test(src_input, trg_input, encoder, decoder, index_to_symbol_trg, test_trg)

    print(f"Bleu score on test set is: {bleu_score}")


if __name__ == "__main__":
    main()
