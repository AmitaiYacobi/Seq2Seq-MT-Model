import os
import json
import numpy as np
import seaborn as sb
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

def symbols_to_indices(raw_data, symbol_to_index):
    return [[symbol_to_index[symbol]  for symbol in seq] for seq in raw_data]

def create_dir(dirname):
    directory = "./" + dirname
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_attention_weights(name, src_example, trg_example, weights):
    fig, ax = plt.subplots(1, figsize=(8, 8), dpi=192)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.set_xticks(np.arange(len(src_example) + 2))
    ax.set_yticks(np.arange(len(src_example) + 1))

    ax = sb.heatmap(weights.numpy(), square=True, cmap="Blues", xticklabels=['begin'] + src_example + ['end'], cbar=False)

    ax.set_yticklabels(trg_example + ['end'], rotation=360)

    ax.set(title=f"attention-based alignment:\n{' '.join(src_example)} -> {' '.join(trg_example)}\n")

    fig.tight_layout()
    plt.savefig("./plots/" + name + ".png", dpi=192, bbox_inches="tight")

def dump_attention_weights(attention_weights, epoch, ):
        f = open('./attention_weights/attentions_weights.json', 'w+', encoding='utf-8')
        weights = [attention.tolist()[0] for attention in attention_weights]
        f.write(f"Epoch {epoch}:\n")
        for weight in weights:
            json.dump(weight, f)
            f.write("\n")
        f.write("\n")
