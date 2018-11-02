import pickle
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import itertools
import functools


def split_label(label_str):
    components = label_str.split('-')
    if len(components) == 2:
        return components[0], components[1]
    assert components[0] == 'O'
    return 'O', 'O'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_corpus', default='../data/ner/eng.train.iobes')
    parser.add_argument('--input_embedding', default="../embedding/glove.6B.100d.txt")
    parser.add_argument('--output_map', default="../data/conll_map.pk")
    parser.add_argument('--threshold', type=int, default=5)
    parser.add_argument('--unk', default='unk')
    args = parser.parse_args()

    gw_map = dict()
    embedding_array = list()
    for line in open(args.input_embedding, 'r'):
        line = line.split()
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        if line[0] == args.unk:
            gw_map['<unk>'] = len(gw_map)
        else:
            gw_map[line[0]] = len(gw_map)
        embedding_array.append(vector)

    bias = 2 * np.sqrt(3.0 / len(embedding_array[0]))

    gw_map['<\n>'] = len(gw_map)
    embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])

    w_count = dict()
    c_count = dict()
    f_map = dict()
    s_map = dict()
    y_map = dict()
    # y_map = {'B-LST':0, 'E-LST':1}

    with open(args.train_corpus, 'r') as fin:
        for line in fin:
            if line.isspace() or line.startswith('-DOCSTART-'):
                c_count['\n'] = c_count.get('\n', 0) + 1
            else:
                line = line.split()
                for tup in line[0]:
                    c_count[tup] = c_count.get(tup, 0) + 1
                c_count[' '] = c_count.get(' ', 0) + 1

                if line[-1] not in y_map:
                    y_map[line[-1]] = len(y_map)

                a, b = split_label(line[-1])
                if a not in f_map:
                    f_map[a] = len(f_map)
                if b not in s_map:
                    s_map[b] = len(s_map)

                word = line[0].lower()
                if word not in gw_map:
                    w_count[word] = w_count.get(word, 0) + 1

    w_map = {v[0]:k for k, v in enumerate(w_count.items()) if v[1] > args.threshold}
    for k in w_map:
        gw_map[k] = len(gw_map)
        embedding_array.append([random.random() * bias - bias for tup in embedding_array[0]])

    filtered_c = [k for k, v in c_count.items() if v > args.threshold]
    c_map = {}
    for i, c in enumerate(filtered_c):
        c_map[c[0]] = i
    c_map['<unk>'] = len(c_map)

    f_map['<eof>'] = len(f_map)
    s_map['<eof>'] = len(s_map)

    y_map['<eof>'] = len(y_map)

    with open(args.output_map, 'wb') as f:
        pickle.dump({
            'gw_map': gw_map, 'c_map': c_map, 'f_map': f_map, 's_map': s_map, 'y_map': y_map, 'emb_array': embedding_array
        }, f)
