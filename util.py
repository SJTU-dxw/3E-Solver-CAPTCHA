import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def sigmoid_rampup(current, rampup_length):
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
        

def get_current_consistency_weight(consistency, consistency_rampup, epoch):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
    

class Seq2SeqLoss(nn.Module):
    def __init__(self):
        super(Seq2SeqLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, y):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        y: [batch_size, max_len]
        """
        max_len = y.size(1)

        loss = sum([self.criterion(outputs[:, i, :], y[:, i + 1]) for i in range(max_len - 1)]) / (max_len - 1)

        return loss


class ConsistentLoss_MT(nn.Module):
    def __init__(self):
        super(ConsistentLoss_MT, self).__init__()

    def forward(self, outputs, outputs_ema):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        max_len = outputs.size(1) + 1

        loss = 0
        for i in range(max_len-1):
            input_logits = outputs[:, i, :]
            target_logits = outputs_ema[:, i, :]
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            loss += F.mse_loss(input_softmax, target_softmax, reduction='none').mean()

        return loss / (max_len - 1)


class ConsistentLoss_MT_Temperature(nn.Module):
    def __init__(self, threshold):
        super(ConsistentLoss_MT_Temperature, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, outputs_ema):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        max_len = outputs.size(1) + 1

        loss = 0
        for i in range(max_len-1):
            input_logits = outputs[:, i, :]
            target_logits = outputs_ema[:, i, :]
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            max_probs, targets_u = torch.max(target_softmax, dim=-1)
            mask = max_probs.ge(self.threshold).float()

            loss += (F.mse_loss(input_softmax, target_softmax, reduction='none').mean(dim=-1) * mask).mean()

        return loss / (max_len - 1)

class ConsistentLoss(nn.Module):
    def __init__(self, threshold):
        super(ConsistentLoss, self).__init__()

        self.threshold = threshold

    def compute_consistent_loss(self, logits_u_w, logits_u_s):
        """
        outputs: [secondary_batch_size, max_len-1, vocab_size]
        outputs_ema: [secondary_batch_size, max_len-1, vocab_size]
        """
        max_len = logits_u_s.size(1) + 1

        pseudo_label = torch.softmax(logits_u_w, dim=-1)
        loss_all = 0
        for i in range(max_len - 1):
            max_probs, targets_u = torch.max(pseudo_label[:, i, :], dim=-1)
            mask = max_probs.ge(self.threshold).float()
            loss_all += (F.cross_entropy(logits_u_s[:, i, :], targets_u,
                                         reduction='none') * mask).mean()

        return loss_all / (max_len - 1)

    def forward(self, logits_u_w, logits_u_s):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        loss = self.compute_consistent_loss(logits_u_w, logits_u_s)

        return loss


def compute_seq_acc(outputs, y, max_len):
    """
    outputs: [batch_size, max_len-1, vocab_size]
    y: [batch_size, max_len]
    """

    accuracy_clevel, accuracy_all = compute_acc_step(outputs, y, max_len)

    return accuracy_clevel, accuracy_all


def compute_acc_step(outputs, y, max_len):
    num_eq = (y[:, 1:].data == outputs.max(2)[1]).sum(dim=1)
    accuracy_clevel = num_eq.sum() / (max_len - 1)
    accuracy_all = (num_eq == max_len - 1).sum()

    return accuracy_clevel.item(), accuracy_all.item()

