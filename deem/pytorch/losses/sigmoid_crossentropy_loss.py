import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result
        """
        logits = y_pred.flatten()
        labels = y_true.flatten()
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        return loss