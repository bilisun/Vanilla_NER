from __future__ import print_function
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math

from model_seq.fully_connected_crf_dataset import FullyConnectedCRFDataset
from model_seq.two_label_evaluator import eval_wc
from model_seq.feature_extractor import FeatureExtractor
from model_seq.TFBase import TFBase
from model_seq.TFCriterion import TFCriterion
import model_seq.utils as utils

from torch_scope import wrapper

import argparse
import json
import os
import sys
import itertools
import functools
import numpy as np


# Hack to save multiple models
class ModelWrapper(nn.Module):
    def __init__(self, extractor, base):
        self.extractor = extractor
        self.base = base


def combine(a, b):
    if a == 'O' and b == 'O':
        return 'O'
    return a + '-' + b


def get_mask(f_map, s_map, y_map):
    f_len = len(f_map)
    s_len = len(s_map)

    mask = np.ones(f_len * s_len)
    for fw, fc in f_map.items():
        for sw, sc in s_map.items():
            w = combine(fw, sw)
            if w not in y_map and not (fw == '<eof>' and sw == '<eof>'):
                mask[s_len * fc + sc] = 0
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--cp_root', default='./checkpoint')
    parser.add_argument('--checkpoint_name', default='ner')
    parser.add_argument('--git_tracking', action='store_true')

    parser.add_argument('--corpus', default='./data/ner_dataset.pk')

    parser.add_argument('--seq_c_dim', type=int, default=30)
    parser.add_argument('--seq_c_hid', type=int, default=150)
    parser.add_argument('--seq_c_layer', type=int, default=1)
    parser.add_argument('--seq_w_dim', type=int, default=100)
    parser.add_argument('--seq_w_hid', type=int, default=300)
    parser.add_argument('--seq_w_layer', type=int, default=1)
    parser.add_argument('--seq_droprate', type=float, default=0.5)
    parser.add_argument('--seq_model', choices=['vanilla'], default='vanilla')
    parser.add_argument('--seq_rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')

    parser.add_argument('--only_unary', dest='only_unary', action='store_true',
                       help='consider only unary potential')
    parser.add_argument('--with_binary', dest='only_unary', action='store_false',
                       help='consider unary + binary potential ')
    parser.add_argument('--no_spatial', dest='no_spatial', action='store_true',
                       help='consider no cross prediction relationship')
    parser.add_argument('--with_spatial', dest='no_spatial', action='store_false',
                       help='considercross prediction relationship')   
    parser.add_argument('--adap', dest='no_adap', action='store_false',
                        help='adaptive message passing')
    parser.add_argument('--no-adap', dest='no_adap', action='store_true',
                        help='no adaptive message passing')
    parser.add_argument('--pairwise_type', default=1, type=int)
    parser.add_argument('--w_temporal', default=1.0, type=float)
    parser.add_argument('--w_spatial', default=1.0, type=float)
    parser.add_argument('--sigma', default=150, type=float)

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--lr_decay', type=float, default=0.05)
    parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta', 'SGD'], default='SGD')
    args = parser.parse_args()

    # automatically sync to spreadsheet
    # pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, enable_git_track=args.git_tracking, \
    #                   sheet_track_name=args.spreadsheet_name, credential_path="/data/work/jingbo/ll2/Torch-Scope/torch-scope-8acf12bee10f.json")
    
    pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, enable_git_track=args.git_tracking)
    pw.set_level('info')

    pw.info('Loading data')

    dataset = pickle.load(open(args.corpus, 'rb'))
    name_list = ['gw_map', 'c_map', 'f_map', 's_map', 'y_map', 'emb_array', 'train_data', 'test_data', 'dev_data']
    gw_map, c_map, f_map, s_map, y_map, emb_array, train_data, test_data, dev_data = [dataset[tup] for tup in name_list]

    
    pw.info('Building models')

    feature_extractor = FeatureExtractor(
            len(c_map), args.seq_c_dim, args.seq_c_hid, args.seq_c_layer, len(gw_map),
            args.seq_w_dim, args.seq_w_hid, args.seq_w_layer, len(y_map), args.seq_droprate,
            unit=args.seq_rnn_unit
    ).cuda()
    feature_extractor.rand_init()
    feature_extractor.load_pretrained_word_embedding(torch.FloatTensor(emb_array))
    seq_config = feature_extractor.to_params()

    fs_mask = get_mask(f_map, s_map, y_map)
    base_model = TFBase(
            len(y_map), len(f_map), len(s_map), fs_mask, no_adap=args.no_adap,
            pairwise_type=args.pairwise_type
    ).cuda()
    base_config = base_model.to_params()

    crit = TFCriterion(
            args.w_temporal, args.w_spatial, args.sigma, args.only_unary, args.no_spatial
    ).cuda()
    evaluator = eval_wc('f1')

    pw.info('Constructing dataset')

    train_dataset, test_dataset, dev_dataset = [
            FullyConnectedCRFDataset(
                tup_data, gw_map['<\n>'], c_map[' '], c_map['\n'], f_map['<eof>'], len(f_map),
                s_map['<eof>'], len(s_map), args.batch_size
            ) for tup_data in [train_data, test_data, dev_data]
    ]

    pw.info('Constructing optimizer')

    all_params = feature_extractor.parameters() + base_model.parameters()
    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9)}
    if args.lr > 0:
        optimizer=optim_map[args.update](all_params, lr=args.lr)
    else:
        optimizer=optim_map[args.update](all_params)

    pw.info('Saving configues.')
    pw.save_configue(args)

    pw.info('Setting up training environ.')
    best_f1 = float('-inf')
    patience_count = 0
    batch_index = 0
    normalizer = 0
    tot_loss = 0

    try:
        for indexs in range(args.epoch):

            pw.info('############')
            pw.info('Epoch: {}'.format(indexs))
            pw.nvidia_memory_map()

            feature_extractor.train()
            base_model.train()
            crit.train()
            for f_c, f_p, b_c, b_p, f_w, label_f, label_s in train_dataset.get_tqdm():

                feature_extractor.zero_grad()
                base_model.zero_grad()
                features = feature_extractor(f_c, f_p, b_c, b_p, f_w)
                f, s, fs, ff, ss, fs_t, sf_t = base_model(features)
                f_out, s_out, loss = crit(f, s, fs, ff, ss, fs_t, sf_t, label_f, label_s)

                tot_loss += utils.to_scalar(loss)
                normalizer += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, args.clip)
                optimizer.step()

                batch_index += 1
                if 0 == batch_index % 100:
                    pw.add_loss_vs_batch({'training_loss': tot_loss / (normalizer + 1e-9)}, batch_index, use_logger = False)
                    tot_loss = 0
                    normalizer = 0

            if args.lr > 0:
                current_lr = args.lr / (1 + (indexs + 1) * args.lr_decay)
                utils.adjust_learning_rate(optimizer, current_lr)

            dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(feature_extractor, base_model,
                    crit, dev_dataset.get_tqdm())

            pw.add_loss_vs_batch({'dev_f1': dev_f1}, indexs, use_logger = True)
            pw.add_loss_vs_batch({'dev_pre': dev_pre, 'dev_rec': dev_rec}, indexs, use_logger = False)
            
            pw.info('Saving model...')
            pw.save_checkpoint(model=ModelWrapper(feature_extractor, base_model),
                        is_best = (dev_f1 > best_f1), 
                        s_dict = {'feature_extractor_config': seq_config, 
                                'tf_base_config': base_config,
                                'best_f1': best_f1,
                                'epoch': indexs})

            if dev_f1 > best_f1:
                test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(feature_extractor,
                        base_model, crit, test_dataset.get_tqdm())
                best_f1, best_dev_pre, best_dev_rec, best_dev_acc = dev_f1, dev_pre, dev_rec, dev_acc
                pw.add_loss_vs_batch({'test_f1': test_f1}, indexs, use_logger = True)
                pw.add_loss_vs_batch({'test_pre': test_pre, 'test_rec': test_rec}, indexs, use_logger = False)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    break

    except Exception as e_ins:

        pw.info('Exiting from training early')

        print(type(e_ins))
        print(e_ins.args)
        print(e_ins)

        dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(feature_extractor, base_model, crit, dev_dataset.get_tqdm())

        pw.add_loss_vs_batch({'dev_f1': dev_f1}, indexs, use_logger = True)
        pw.add_loss_vs_batch({'dev_pre': dev_pre, 'dev_rec': dev_rec}, indexs, use_logger = False)
        
        pw.info('Saving model...')
        pw.save_checkpoint(model=ModelWrapper(feature_extractor, base_model),
                    is_best = (dev_f1 > best_f1), 
                    s_dict = {'feature_extractor_config': seq_config,
                            'tf_base_config': base_config,
                            'best_f1': best_f1,
                            'epoch': indexs})

        test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(feature_extractor, base_model,
                crit, test_dataset.get_tqdm())
        best_f1, best_dev_pre, best_dev_rec, best_dev_acc = dev_f1, dev_pre, dev_rec, dev_acc
        pw.add_loss_vs_batch({'test_f1': test_f1}, indexs, use_logger = True)
        pw.add_loss_vs_batch({'test_pre': test_pre, 'test_rec': test_rec}, indexs, use_logger = False)

    pw.close()
