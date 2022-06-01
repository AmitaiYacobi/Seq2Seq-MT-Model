import os
import matplotlib.pyplot as plt

BEGIN_SYMBOL = '<s>'
END_SYMBOL = '</s>'

def read_data(src_file, trg_file):
    src_data = open(src_file, 'r')
    trg_data = open(trg_file, 'r')

    src = [[BEGIN_SYMBOL] + line.strip().split() + [END_SYMBOL] for line in src_data]
    trg = [[BEGIN_SYMBOL] + line.strip().split() + [END_SYMBOL] for line in trg_data]

    return src, trg

def create_vocabulary(train, dev, test):
    vocabulary = set()

    for line in train:
        for symbol in line:
            vocabulary.add(symbol)
    
    for line in dev:
        for symbol in line:
            vocabulary.add(symbol)
    
    for line in test:
        for symbol in line:
            vocabulary.add(symbol)

    return sorted(vocabulary)

def map_symbols_and_indices(vocabulary):
    symbol_to_index = {word: i for i, word in enumerate(vocabulary)}
    index_to_symbol = {i: word for i, word in enumerate(vocabulary)}
    return symbol_to_index, index_to_symbol

def symbol_to_indices(raw_data, symbol_to_index):
    return [[symbol_to_index[symbol]  for symbol in seq] for seq in raw_data]

def create_dir(dirname):
    directory = "./" + dirname
    if not os.path.exists(directory):
        os.makedirs(directory)
