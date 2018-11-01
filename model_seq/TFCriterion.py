import torch
import torch.nn as nn
from torch.autograd import Variable
import math


def gaussian_kernel(t1, t2, sigma):
    return math.exp(-(float(t1 - t2) * (t1 - t2)) / (2 * self.sigma * self.sigma))


class MessagePassing(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def get_past_messages(self, potentials, seq_len, batch_size, classes):
        """
        Get averaged past message for each (position, sequence number) pair
        Average rather than sum to balance pairwise and unary potentials

        potentials: (sequence length * batch size, classes)
        output: (sequence length * batch_size, classes)
        """

        messages = []
        for s in range(seq_len):
            for b in range(batch_size):
                if s == 0:
                    messages.append(torch.zeros(classes))
                else:
                    avg_msg = potentials[b].clone() * gaussian_kernel(0, s, self.sigma)
                    for i in range(1, s):
                        avg_msg += (potentials[batch_size * i + b] *
                                    gaussian_kernel(i, s, self.sigma))
                    avg_msg /= s
                    messages.append(avg_msg)

        return Variable(torch.stack(messages).cuda())

    def get_future_messages(self, potentials, seq_len, batch_size, classes):
        """
        Get averaged future message for each (position, sequence number) pair
        Average rather than sum to balance pairwise and unary potentials

        potentials: (sequence length * batch size, classes)
        output: (sequence length * batch_size, classes)
        """

        messages = []
        for s in range(seq_len):
            for b in range(batch_size):
                if s == seq_len - 1:
                    messages.append(torch.zeros(classes))
                else:
                    avg_msg = (potentials[batch_size * (s + 1) + b].clone() *
                               gaussian_kernel(s, s + 1, self.sigma))
                    for i in range(s + 2, seq_len):
                        avg_msg += (potentials[batch_size * i + b] *
                                    gaussian_kernel(s, i, self.sigma))
                    avg_msg /= (seq_len - s - 1)
                    messages.append(avg_msg)

        return Variable(torch.stack(messages).cuda())


class TFCriterion(nn.Module, MessagePassing):
    """
    Implements mean field approximation message passing
    """
    def __init__(self, w_temporal, w_spatial, sigma, only_unary, no_spatial):
        MessagePassing.__init__(self, sigma)
        nn.Module.__init__(self)

        self.train_mp_iterations = 5

        self.w_temporal = w_temporal
        self.w_spatial = w_spatial
        self.only_unary = only_unary
        self.no_spatial = no_spatial

        self.f_loss = nn.CrossEntropyLoss()
        self.s_loss = nn.CrossEntropyLoss()

    def forward(self, f, s, fs, ff, ss, fs_t, sf_t, f_labels, s_labels, evaluate=False):
        """
        f: (sequence length, batch size, f classes),
        s: (sequence length, batch size, s classes),
        fs: (sequence length, batch size, f classes, s classes)
        ff: (sequence length, batch size, f classes, f classes)
        ss: (sequence length, batch size, s classes, s classes)
        fs_t: (sequence length, batch size, f classes, s classes)
        sf_t: (sequence length, batch size, s classes, f classes)

        f_labels: (batch size, sequence length)
        s_labels: (batch size, sequence length)
        evaluate: If True, multiple message passing iterations, otherwise 1 for training

        f_out: (sequence length, batch size, f classes)
        s_out: (sequence length, batch size, s classes)
        loss: Training loss
        """

        num_iterations = 1
        if evaluate:
            num_iterations = self.train_mp_iterations

        seq_len = f.shape[0]
        batch_size = f.shape[1]
        f_classes = f.shape[2]
        s_classes = s.shape[2]

        # Flatten all input tensors to (sequence length * batch, ...)

        orig_f = f.view(-1, f_classes)
        orig_s = s.view(-1, s_classes)

        prev_f = orig_f.clone()
        prev_s = orig_s.clone()

        loss = self.f_loss(orig_f, f_labels) + self.s_loss(orig_s, s_labels)

        flat_fs = fs.view(-1, f_classes, s_classes)
        flat_ff = ff.view(-1, f_classes, f_classes)
        flat_ss = ss.view(-1, s_classes, s_classes)
        flat_fs_t = fs_t.view(-1, f_classes, s_classes)
        flat_sf_t = sf_t.view(-1, s_classes, f_classes)

        # Message passing iterations
        if not self.only_unary:
            for iteration in range(num_iterations):
                f_msg_past = self.get_past_messages(prev_f, seq_len, batch_size, f_classes)
                f_msg_future = self.get_future_messages(prev_f, seq_len, batch_size, f_classes)
                s_msg_past = self.get_past_messages(prev_s, seq_len, batch_size, s_classes)
                s_msg_future = self.get_future_messages(prev_s, seq_len, batch_size, s_classes)

                next_f = orig_f.clone()
                next_s = orig_s.clone()

                # Message passing update from prev_f / prev_s -> next_f / next_s
                next_f += torch.bmm(f_msg_past.unsqueeze(1), flat_ff).squeeze() * self.w_temporal
                next_f += torch.bmm(flat_ff, f_msg_future.unsqueeze(2)).squeeze() * self.w_temporal

                next_s += torch.bmm(s_msg_past.unsqueeze(1), flat_ss).squeeze() * self.w_temporal
                next_s += torch.bmm(flat_ss, s_msg_future.unsqueeze(2)).squeeze() * self.w_temporal

                if not self.no_spatial:
                    next_f += torch.bmm(s_msg_past.unsqueeze(1), flat_sf_t).squeeze() * self.w_temporal
                    next_f += torch.bmm(flat_fs_t, s_msg_future.unsqueeze(2)).squeeze() * self.w_temporal
                    next_f += torch.bmm(flat_fs, prev_s.unsqueeze(2)).squeeze() * self.w_spatial

                    next_s += torch.bmm(f_msg_past.unsqueeze(1), flat_fs_t).squeeze() * self.w_temporal
                    next_s += torch.bmm(flat_sf_t, f_msg_future.unsqueeze(2)).squeeze() * self.w_temporal
                    next_s += torch.bmm(prev_f.unsqueeze(1), flat_fs).squeeze() * self.w_spatial

                # Update message passing memory
                prev_f = nn.Softmax(dim=1)(next_f)
                prev_s = nn.Softmax(dim=1)(next_s)

            loss += self.f_loss(next_f, f_labels)
            loss += self.s_loss(next_s, s_labels)

        f_out = prev_f.view(seq_len, -1, f_classes)
        s_out = prev_s.view(seq_len, -1, s_classes)
        return f_out, s_out, loss
