import torch.nn as nn
import torch
import numpy as np


class PairwiseMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(PairwiseMarginLoss, self).__init__()
        self._margin = margin

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        loss = torch.relu(self._margin + neg_logits - pos_logits).mean()
        return loss