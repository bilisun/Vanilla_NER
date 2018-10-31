"""
Based on Liyuan Liu's SeqDataset code
"""
import torch
from torch.autograd import Variable

import numpy as np
import random
from tqdm import tqdm
import sys
import itertools


def to_onehot(indices, size):
    onehot = np.zeros((len(indices), size))
    onehot[np.arange(len(indices)), indices] = 1
    return onehot


class FullyConnectedCRFDataset(object):
    """
    Dataset for fully connected CRF

    Shorter sequences are padded to the max length in the batch
    """
    def __init__(
            self, dataset: list, w_pad: int, c_con: int, c_pad: int, f_pad: int,
            f_size: int, s_pad: int, s_size: int, batch_size: int, f_classes: int,
            s_classes: int
    ):
        self.w_pad = w_pad
        self.c_con = c_con
        self.c_pad = c_pad
        self.f_pad = f_pad
        self.f_size = f_size
        self.s_pad = s_pad
        self.s_size = s_size
        self.batch_size = batch_size

        self.f_classes = f_classes
        self.s_classes = s_classes

        self.construct_index(dataset)
        self.shuffle()

    def shuffle(self):
        """
        shuffle dataset
        """
        random.shuffle(self.shuffle_list)

    def get_tqdm(self):
        """
        construct dataset reader and the corresponding tqdm.
        """
        return tqdm(self.reader(), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def construct_index(self, dataset):
        """
        construct index for the dataset.

        Parameters
        ----------
        dataset: ``list``, required.
            the encoded dataset (outputs of preprocess scripts).
        """
        for instance in dataset:
            c_len = [len(tup)+1 for tup in instance[1]]
            c_ins = [tup for ins in instance[1] for tup in (ins + [self.c_con])]
            instance[1] = c_ins
            instance.append(c_len)

        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def reader(self):
        """
        construct dataset reader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object
        """
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch)
        self.shuffle()

    def batchify(self, batch):
        """
        batchify a batch of data and move to gpu.

        Parameters
        ----------
        batch: ``list``, required.
            a sample from the encoded dataset (outputs of preprocess scripts).  

        Returns
        ----------
        (forward character ids,
         forward character indices,
         backward character ids,
         backward character indices,
         word ids,
         label 1 onehot,
         label 2 onehot,
         raw label 1,
         raw label 2)
        """

        cur_batch_size = len(batch)

        char_padded_len = max([len(tup[1]) for tup in batch])
        word_padded_len = max([len(tup[0]) for tup in batch])

        tmp_batch =  [list() for ind in range(9)]

        for instance_ind in range(cur_batch_size):
            instance = batch[instance_ind]

            char_padded_len_ins = char_padded_len - len(instance[1])
            word_padded_len_ins = word_padded_len - len(instance[0])

            tmp_batch[0].append(instance[1] + [self.c_pad] * char_padded_len_ins)
            tmp_batch[2].append(instance[1][::-1] + [self.c_pad] * char_padded_len_ins)

            tmp_pf = list(itertools.accumulate(instance[4] + [0] * word_padded_len_ins))
            tmp_pb = list(itertools.accumulate(instance[4][::-1]))[::-1] + [1] * word_padded_len_ins

            tmp_batch[1].append(
                    [(word_end - 1) * cur_batch_size + instance_ind for word_end in tmp_pf]
            )
            tmp_batch[3].append(
                    [(word_end - 1) * cur_batch_size + instance_ind for word_end in tmp_pb]
            )

            tmp_batch[4].append(instance[0] + [self.w_pad] * word_padded_len_ins)
            tmp_batch[5].append(
                    to_onehot(instance[2] + [self.f_pad] * word_padded_len_ins, self.f_classes)
            )
            tmp_batch[6].append(
                    to_onehot(instance[3] + [self.s_pad] * word_padded_len_ins, self.s_classes)
            )

            tmp_batch[7].append(instance[2])
            tmp_batch[8].append(instance[3])

        # Tensor shapes are now (number of chars or words, batch size)
        tbt = [torch.LongTensor(v).transpose(0, 1).contiguous() for v in tmp_batch[0:7]]

        tbt[1] = tbt[1].view(-1)
        tbt[3] = tbt[3].view(-1)

        return [Variable(ten.cuda()) for ten in tbt] + [tmp_batch[7], tmp_batch[8]]
