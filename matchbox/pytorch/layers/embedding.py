import torch
from torch import nn
import h5py
import os
import numpy as np
from collections import OrderedDict
from .. import layers


class EmbeddingLayer(nn.Module):
    def __init__(self, 
                 feature_map,
                 embedding_dim,
                 disable_sharing_pretrain=False,
                 required_feature_columns=[],
                 not_required_feature_columns=[]):
        super(EmbeddingLayer, self).__init__()
        self.embedding_layer = EmbeddingDictLayer(feature_map, 
                                                  embedding_dim,
                                                  disable_sharing_pretrain=disable_sharing_pretrain,
                                                  required_feature_columns=required_feature_columns,
                                                  not_required_feature_columns=not_required_feature_columns)

    def forward(self, X, feature_source=None):
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        return feature_emb


class EmbeddingDictLayer(nn.Module):
    def __init__(self, 
                 feature_map, 
                 embedding_dim,
                 disable_sharing_pretrain=False,
                 required_feature_columns=None,
                 not_required_feature_columns=None):
        super(EmbeddingDictLayer, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.embedding_layers = nn.ModuleDict()
        self.embedding_callbacks = nn.ModuleDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if self.is_required(feature):
                if disable_sharing_pretrain: # in case for LR
                    assert embedding_dim == 1
                    feat_emb_dim = embedding_dim
                else:
                    feat_emb_dim = feature_spec.get("embedding_dim", embedding_dim)
                if (not disable_sharing_pretrain) and "embedding_callback" in feature_spec:
                    self.embedding_callbacks[feature] = eval(feature_spec["embedding_callback"])
                # Set embedding_layer according to share_embedding
                if (not disable_sharing_pretrain) and "share_embedding" in feature_spec:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue
                    
                if feature_spec["type"] == "numeric":
                    self.embedding_layers[feature] = nn.Linear(1, feat_emb_dim, bias=False)
                elif feature_spec["type"] == "categorical":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                    feat_emb_dim,
                                                    padding_idx=padding_idx)
                    if (not disable_sharing_pretrain) and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix,
                                                                          feature_map, 
                                                                          feature_name, 
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
                elif feature_spec["type"] == "sequence":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = nn.Embedding(feature_spec["vocab_size"], 
                                                    feat_emb_dim, 
                                                    padding_idx=padding_idx)
                    if (not disable_sharing_pretrain) and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix, 
                                                                          feature_map, 
                                                                          feature_name,
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix

    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.feature_specs[feature]
        if self.required_feature_columns and (feature not in self.required_feature_columns):
            return False
        if self.not_required_feature_columns and (feature in self.not_required_feature_columns):
            return False
        return True

    def get_pretrained_embedding(self, pretrained_path, feature_name):
        with h5py.File(pretrained_path, 'r') as hf:
            embeddings = hf[feature_name][:]
        return embeddings

    def load_pretrained_embedding(self, embedding_matrix, feature_map, feature_name, freeze=False, padding_idx=None):
        pretrained_path = os.path.join(feature_map.data_dir, feature_map.feature_specs[feature_name]["pretrained_emb"])
        embeddings = self.get_pretrained_embedding(pretrained_path, feature_name)
        if padding_idx is not None:
            embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
        embeddings = torch.from_numpy(embeddings).float()
        embedding_matrix.weight = torch.nn.Parameter(embeddings)
        if freeze:
            embedding_matrix.weight.requires_grad = False
        return embedding_matrix

    def dict2tensor(self, embedding_dict):
        if len(embedding_dict) == 1:
            feature_emb = list(embedding_dict.values())[0]
        else:
            feature_emb = torch.stack(list(embedding_dict.values()), dim=1)
        return feature_emb

    def forward(self, inputs, feature_source=None, feature_type=None):
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if feature_source and feature_spec["source"] != feature_source:
                continue
            if feature_type and feature_spec["type"] != feature_type:
                continue
            if feature in self.embedding_layers:
                if feature_spec["type"] == "numeric":
                    inp = inputs[feature].float().view(-1, 1)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.embedding_callbacks:
                    embeddings = self.embedding_callbacks[feature](embeddings)     
                feature_emb_dict[feature] = embeddings
        return feature_emb_dict




