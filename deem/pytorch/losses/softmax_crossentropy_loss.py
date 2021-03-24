import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        probs = F.softmax(y_pred, dim=1)
        hit_probs = probs[:, 0]
        loss = -torch.log(hit_probs).mean()
        return loss