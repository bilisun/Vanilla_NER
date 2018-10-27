"""
Adaptive Asynchronous Temporal Fields Base model
"""

import torch.nn as nn
import torch
from torch.autograd import Variable


class BasicModule(nn.Module):
    def __init__(self, in_dim, out_dim, _type=2, hidden_dim=1000, droprate=0.3):
        super(BasicModule, self).__init__()

        if _type == 1:
            self.layers = nn.Linear(in_dim, out_dim, bias=False)
        elif _type == 2:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=droprate),
                nn.Linear(hidden_dim, out_dim)
            )

    def forward(self, x):
        return self.layers(x)


class TFBase(nn.Module):
    def __init__(self, dim, num_classes, num_low_rank, no_adap=True, pairwise_type=1, _BaseModule = BasicModule):
        super(TFBase, self).__init__()

        self.num_classes = num_classes
        self.num_low_rank = num_low_rank
        self.no_adap = no_adap

        # Unary potentials
        self.l = nn.Linear(dim, self.c_classes)

        # Temporal label compatibility matrices
        self.ll_a = (
                _BaseModule(1, self.num_classes * self.num_low_rank, _type = 1) if self.no_adap else
                _BaseModule(dim, self.num_classes * self.num_low_rank, _type = pairwise_type)
        )
        self.ll_b = (
                _BaseModule(1, self.num_low_rank * self.num_classes, _type = 1) if self.no_adap else
                _BaseModule(dim, self.num_low_rank * self.num_classes, _type = pairwise_type)
        )

    def forward(self, features):
        """
        features: (sequence length, batch size, hidden dim)

        output: (sequence length, batch size, classes),
                (sequence length, batch size, classes, classes)
        """

        hidden_dim = features.shape[2]
        seq_features = features.view(-1, hidden_dim)

        l = self.l(seq_features)
        vl = l.view(-1, features.shape[1], self.num_classes)

        adapted_features = (
                Variable(torch.ones(seq_features.shape[0], 1).cuda())
                if self.no_adap else seq_features
        )

        ll_a = self.ll_a(feat).view(-1, self.num_classes, self.num_low_rank)
        ll_b = self.ll_b(feat).view(-1, self.num_low_rank, self.num_classes)
        ll = torch.bmm(ll_a, ll_b)
        vll = ll.view(-1, features.shape[1], self.num_classes, self.num_classes)

        return vl, vll
