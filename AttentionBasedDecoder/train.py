import sys
import matplotlib.pyplot as plt

from utils import *
from Model import *

def main():
    train_src_file = sys.argv[1]
    train_trg_file = sys.argv[2]
    dev_src_file = sys.argv[3]
    dev_trg_file = sys.argv[4]
    test_src_file = sys.argv[5]
    test_trg_file = sys.argv[6]
    
    train_src, train_trg = read_data(train_src_file, train_trg_file)
    dev_src, dev_trg = read_data(dev_src_file, dev_trg_file)
    test_src, test_trg = read_data(test_src_file, test_trg_file)

    src_vocab = create_vocabulary(train_src, dev_src, test_src)
    trg_vocab = create_vocabulary(train_trg, dev_trg, test_trg)

    symbol_to_index_src, index_to_symbol_src = map_symbols_and_indices(src_vocab)
    symbol_to_index_trg, index_to_symbol_trg = map_symbols_and_indices(trg_vocab)
    
    train_src_indices = symbols_to_indices(train_src, symbol_to_index_src)
    train_trg_indices = symbols_to_indices(train_trg, symbol_to_index_trg) 

    dev_src_indices = symbols_to_indices(dev_src, symbol_to_index_src)
    dev_trg_indices = symbols_to_indices(dev_trg, symbol_to_index_trg)
    
    create_dir("results")
    model = Seq2SeqModel(
        train_src_indices,
        train_trg_indices,
        dev_src_indices,
        dev_trg_indices,
        symbol_to_index_src,
        symbol_to_index_trg,
        dev_trg, 
        index_to_symbol_trg
    )
    
    train_losses, dev_losses, bleu_scores = model.train()
    
    epochs = [int(i) for i in range(len(bleu_scores))]

    plt.title("Blue scores")
    plt.plot(epochs, bleu_scores, color='blue')
    plt.ylabel("Score")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.show()

    plt.title("Loss")
    plt.plot(epochs, train_losses, color='green', label='train loss')
    plt.plot(epochs, dev_losses, color='blue', label='dev loss')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.legend()
    plt.show()

    create_dir("vocabs")
    torch.save({
        'src_vocab' : src_vocab,
        'trg_vocab' : trg_vocab
    }, "./vocabs/vocabs")

if __name__ == "__main__":
    main()