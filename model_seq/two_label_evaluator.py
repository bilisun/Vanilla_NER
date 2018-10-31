import torch
import numpy as np
import itertools

import model_seq.utils as utils
from model_seq.seq_utils import combine, symb_seq_to_spans


class eval_batch:
    """
    Base class for evaluation, provide method to calculate f1 score and accuracy.

    Parameters
    ----------
    decoder : ``torch.nn.Module``, required.
        the decoder module, which needs to contain the ``to_span()`` method.
    """
    def __init__(self, valid_label_mask, f_map, s_map):
        self.valid_label_mask = valid_label_mask
        self.rev_f_map = {v: k for k, v in f_map.items()}
        self.rev_s_map = {v: k for k, v in s_map.items()}

    def reset(self):
        """
        reset counters.
        """
        self.correct_labels = 0
        self.total_labels = 0
        self.actual_positives = 0
        self.predicted_positives = 0
        self.true_positives = 0

    def calc_f1_batch(self, f_out, s_out, raw_label_f, raw_label_s):
        """
        update statics for f1 score.

        f_out: (max seq len, batch size, f_classes)
        s_out: (max seq len, batch size, s_classes)
        raw_label_f: (batch size, varying seq lens)
        raw_label_s: (batch size, varying seq lens)
        """

        batches = f_out.shape[1]

        for seq_num in range(batches):
            correct_labels_i, total_labels_i, actual_positives_i, predicted_positives_i, true_positives_i = self.eval_instance(
                    f_out[:,seq_num,:], s_out[:,seq_num,:], raw_label_f[seq_num], raw_label_s[seq_num]
            )
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.actual_positives += actual_positives_i
            self.predicted_positives += predicted_positives_i
            self.true_positives += true_positives_i

    def calc_acc_batch(self, f_out, s_out, raw_label_f, raw_label_s):
        """
        update statics for accuracy score.

        f_out: (max seq len, batch size, f_classes)
        s_out: (max seq len, batch size, f_classes)
        raw_label_f: (batch size, varying seq lens)
        raw_label_s: (batch size, varying seq lens)
        """

        batches = f_out.shape[1]

        for seq_num in range(batches):
            correct_labels_i, total_labels_i, _, _, _ = self.eval_instance(
                    f_out[:,seq_num,:], s_out[:,seq_num,:], raw_label_f[seq_num], raw_label_s[seq_num]
            )
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i

    def f1_score(self):
        """
        calculate the f1 score based on the inner counter.
        """
        if self.predicted_positives == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = self.true_positives / float(self.predicted_positives)
        recall = self.true_positives / float(self.actual_positives)
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        return f, precision, recall, accuracy

    def acc_score(self):
        """
        calculate the accuracy score based on the inner counter.
        """
        if self.total_labels == 0:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy

    def eval_instance(self, f, s, fl, sl):
        """
        Calculate statistics to update inner counters for one sequence

        f: (max seq length, f_classes), sequence probabilities
        s: (max seq length, s_classes), sequence probabilities
        fl: (max seq length), raw labels
        sl: (max seq length), raw labels
        """

        seq_len = len(fl)
        f_classes = f.shape[1]
        s_classes = s.shape[1]

        symb_seq = []
        expected_symb_seq = []

        total_labels = seq_len
        correct_labels = 0

        for i in range(seq_len):
            best_f = -1
            best_s = -1
            best_p = -1.0
            for fi in range(f_classes):
                for si in range(s_classes):
                    if (self.valid_label_mask[fi * s_classes + si] == 1 and
                        best_p < f[i][fi] * s[i][si]):
                        best_f = fi
                        best_s = si
                        best_p = f[i][fi] * s[i][si]

            symb_seq.append(combine(self.rev_f_map[best_f], self.rev_s_map[best_s]))
            expected_symb_seq.append(combine(self.rev_f_map[fl[i]], self.rev_s_map[sl[i]]))

            if symb_seq[-1] == expected_symb_seq[-1]:
                correct_labels += 1

        predicted_spans = symb_seq_to_spans(symb_seq)
        expected_spans = symb_seq_to_spans(expected_symb_seq)

        actual_positives = len(expected_spans)
        predicted_positives = len(predicted_spans)
        true_positives = len(predicted_spans & expected_spans)

        return correct_labels, total_labels, actual_positives, predicted_positives, true_positives


class eval_wc(eval_batch):
    """
    evaluation class for LD-Net

    Parameters
    ----------
    score_type : ``str``, required.
        whether the f1 score or the accuracy is needed.
    """
    def __init__(self, score_type, valid_label_mask, f_map, s_map):
        eval_batch.__init__(self, valid_label_mask, f_map, s_map)

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, feature_extractor, base_model, crit, dataset_loader):
        """
        calculate scores

        Returns
        -------
        score: ``float``.
            calculated score.
        """
        feature_extractor.eval()
        base_model.eval()
        crit.eval()
        self.reset()

        for f_c, f_p, b_c, b_p, f_w, label_f, label_s, raw_label_f, raw_label_s in dataset_loader:
            features = feature_extractor(f_c, f_p, b_c, b_p, f_w)
            f, s, fs, ff, ss, fs_t, sf_t = base_model(features)
            f_out, s_out, _ = crit(f, s, fs, ff, ss, fs_t, sf_t, label_f, label_s)
            self.eval_b(f_out, s_out, raw_label_f, raw_label_s)

        return self.calc_s()
