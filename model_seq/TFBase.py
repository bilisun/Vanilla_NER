"""
Adaptive Asynchronous Temporal Fields Base model
"""

import torch.nn as nn
import torch
from torch.autograd import Variable


class BasicModule(nn.Module):
    def __init__(self, in_dim, a_classes, b_classes, mask=None, _type=2, hidden_dim=1000, droprate=0.3):
        super(BasicModule, self).__init__()

        self.a_classes = a_classes
        self.b_classes = b_classes
        out_dim = a_classes * b_classes

        self.mask = mask

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
        output = self.layers(x)

        if self.mask is not None:
            output = output * self.mask

        return output.view(-1, self.a_classes, self.b_classes)


class TFBase(nn.Module):
    def __init__(self, dim, f_classes, s_classes, fs_mask, no_adap=True, pairwise_type=1, _BaseModule=BasicModule):
        super(TFBase, self).__init__()

        self.f_classes = f_classes
        self.s_classes = s_classes
        self.no_adap = no_adap
        self.pairwise_type = pairwise_type

        # Unary potentials
        self.f = nn.Linear(dim, self.f_classes)
        self.s = nn.Linear(dim, self.s_classes)

        mask = Variable(torch.Tensor(fs_mask).cuda())
        mask_t = mask.view(-1, self.s_classes).t().view(-1)

        # Spatio label compatibility matrices
        self.fs_w = (
                _BaseModule(1, self.f_classes, self.s_classes, mask=mask, _type=1) if self.no_adap else
                _BaseModule(dim, self.f_classes, self.s_classes, mask=mask, _type=pairwise_type)
        )

        # Temporal label compatibility matrices
        self.ff_w = (
                _BaseModule(1, self.f_classes, self.f_classes, _type=1) if self.no_adap else
                _BaseModule(dim, self.f_classes, self.f_classes, _type=pairwise_type)
        )
        self.ss_w = (
                _BaseModule(1, self.s_classes, self.s_classes, _type=1) if self.no_adap else
                _BaseModule(dim, self.s_classes, self.s_classes, _type=pairwise_type)
        )
        self.fs_t_w = (
                _BaseModule(1, self.f_classes, self.s_classes, mask=mask, _type=1) if self.no_adap else
                _BaseModule(dim, self.f_classes, self.s_classes, mask=mask, _type=pairwise_type)
        )
        self.sf_t_w = (
                _BaseModule(1, self.s_classes, self.f_classes, mask=mask_t, _type=1) if self.no_adap else
                _BaseModule(dim, self.s_classes, self.f_classes, mask=mask_t, _type=pairwise_type)
        )

    def to_params(self):
        return {'adap': self.no_adap, 'pairwise_type': self.pairwise_type}

    def forward(self, features):
        """
        features: (sequence length, batch size, hidden dim)

        f: (sequence length, batch size, f classes),
        s: (sequence length, batch size, s classes),
        fs: (sequence length, batch size, f classes, s classes)
        ff: (sequence length, batch size, f classes, f classes)
        ss: (sequence length, batch size, s classes, s classes)
        fs_t: (sequence length, batch size, f classes, s classes)
        sf_t: (sequence length, batch size, s classes, f classes)
        """

        seq_length = features.shape[1]
        hidden_dim = features.shape[2]
        seq_features = features.view(-1, hidden_dim)

        adapted_features = (
                Variable(torch.ones(seq_features.shape[0], 1).cuda())
                if self.no_adap else seq_features
        )

        flat_f = self.f(seq_features)
        flat_s = self.s(seq_features)

        flat_fs = self.fs_w(adapted_features)

        flat_ff = self.ff_w(adapted_features)
        flat_ss = self.ss_w(adapted_features)
        flat_fs_t = self.fs_t_w(adapted_features)
        flat_sf_t = self.sf_t_w(adapted_features)

        f = flat_f.view(seq_length, -1, self.f_classes)
        s = flat_s.view(seq_length, -1, self.s_classes)

        fs = flat_fs.view(seq_length, -1, self.f_classes, self.s_classes)

        ff = flat_ff.view(seq_length, -1, self.f_classes, self.f_classes)
        ss = flat_ss.view(seq_length, -1, self.s_classes, self.s_classes)
        fs_t = flat_fs_t.view(seq_length, -1, self.f_classes, self.s_classes)
        sf_t = flat_sf_t.view(seq_length, -1, self.s_classes, self.f_classes)

        return f, s, fs, ff, ss, fs_t, sf_t
