from torch import nn
import torch

class MaskedAveragePooling(nn.Module):
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix):
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1)
        non_padding_length = (embedding_matrix.sum(dim=-1) != 0).sum(dim=1, keepdim=True)
        embedding_vec = sum_pooling_matrix / (non_padding_length.float() + 1.e-12)
        return embedding_vec


class MaskedSumPooling(nn.Module):
    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        return torch.sum(embedding_matrix, dim=1)