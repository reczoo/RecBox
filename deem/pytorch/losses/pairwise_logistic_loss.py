import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class PairwiseLogisticLoss(nn.Module):
    def __init__(self):
        super(PairwiseLogisticLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = y_pred[:, 0].unsqueeze(-1)
        neg_logits = y_pred[:, 1:]
        logits_diff = pos_logits - neg_logits
        loss = -torch.log(torch.sigmoid(logits_diff)).mean()
        return loss