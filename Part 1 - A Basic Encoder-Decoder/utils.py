import os

BEGIN_SYMBOL = '<s>'
END_SYMBOL = '</s>'

def read_data(src_file, trg_file):

    src = open(src_file, "r", encoding="utf-8")
    trg = open(trg_file, "r", encoding="utf-8")

    s_data = src.readlines()
    t_data = trg.readlines()

    # train_data = [tuple([begin_symbol] + line.strip().split() + [end_symbol] for line in pair) for pair in zip(f_data, e_data)]
    src = [[BEGIN_SYMBOL] + line.strip().split() + [END_SYMBOL] for line in s_data]
    trg = [[BEGIN_SYMBOL] + line.strip().split() + [END_SYMBOL] for line in s_data]

    return src, trg

def create_vocabulary(data):
    
    vocabulary = set()
    for line in data:
        for symbol in line:
            vocabulary.add(symbol)
    
    return sorted(vocabulary)
    
def symbols_to_indices(vocabulary):
    return {symbol:i for i, symbol in enumerate(vocabulary)}
    
def indices_to_symbols(vocabulary):
    return {i:symbol for symbol, i in enumerate(vocabulary)}

def raw_data_to_indices(data, symbol_to_index):
    return [[symbol_to_index[symbol] for symbol in seq] for seq in data]

def prepare_data(src_file, trg_file):
    src, trg = read_data(src_file, trg_file)
    
    src_vocab = create_vocabulary(src)
    trg_vocab = create_vocabulary(trg)
    
    src_symbol_to_index = symbols_to_indices(src_vocab)
    trg_symbol_to_index = symbols_to_indices(trg_vocab)
    
    src_index_to_symbol = indices_to_symbols(src_vocab)
    trg_index_to_symbol = indices_to_symbols(trg_vocab)
    
    src_indices = raw_data_to_indices(src, src_symbol_to_index)
    trg_indices = raw_data_to_indices(trg, trg_symbol_to_index)
    
    return src_indices, trg_indices, src_vocab, trg_vocab, trg
    
    
# read_data("../data/train.src", "../data/train.trg")