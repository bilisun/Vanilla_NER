import torch
import torch.nn as nn
from torch.autograd import Variable
import math


def weighted_average(iterator, weight=1.0):
    """
    Compounded weighted average
    """

    item, w = next(iterator)
    total = item.clone() * w
    n = 1.0
    for i, (item, w) in enumerate(iterator):
        w1 = 1.0 * weight ** (i + 1)
        total += item * w1 * w
        n += w1
    return total / n


class MessagePassing(object):
    def __init__(self, w_temporal, memory_decay, sigma):

    def get_forward_messages(self, pairs):
        """
        pairs: (sequence length, batch size, classes)
        """

    def get_backward_messages(self, pairs):


class TFCriterion(nn.Module, MessagePassing):
    """
    Implements mean field approximation message passing
    """
    def __init__(self, w_temporal, memory_decay, sigma, only_unary):
        MessagePassing.__init__(self, w_temporal, memory_decay, sigma)
        nn.Module.__init__(self)

        self.train_mp_iterations = 5
        self.only_unary = only_unary

    def forward(self, unaries, pairs, labels, evaluate=False):
        """
        unaries: (sequence length, batch size, classes)
        pairs: (sequence length, batch size, classes, classes)
        labels: (sequence length, batch size, classes)
        evaluate: If True, multiple message passing iterations, otherwise 1 for training

        output: probabilities (sequence length, batch size, classes), loss
        """

        num_iterations = 1
        if evaluate:
            num_iterations = self.train_mp_iterations

        seq_len = unaries.shape[0]
        batch_size = unaries.shape[1]
        num_classes = unaries.shape[2]

        # (sequence length * batch, classes)
        reshaped_unaries = unaries.view(-1, num_classes)
        probabilities = reshaped_unaries.copy()

        reshaped_labels = labels.view(-1, num_classes)

        loss = nn.CrossEntropyLoss()(reshaped_unaries, reshaped_labels)

        # (sequence length * batch, classes, classes)
        reshaped_pairs = pairs.view(-1, num_classes, num_classes)

        if not self.only_unary:
            # Message passing iterations
            for iteration in range(num_iterations):
                f_msg = self.get_forward_messages(pairs, probabilities).view(
                        -1, num_classes, num_classes
                )
                b_msg = self.get_backward_messages(pairs, probabilities).view(
                        -1, num_classes, num_classes
                )

            loss += nn.CrossEntropyLoss()(probabilities, reshaped_labels)

        return probabilities, loss
