from torch import nn
import torch
from .base_model import BaseModel
from ..layers import EmbeddingDictLayer, EmbeddingLayer
import torch.nn.functional as F
import numpy as np


class SimpleX(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="SimpleX", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 user_id_field="user_id",
                 item_id_field="item_id",
                 user_history_field="user_history",
                 enable_bias=False,
                 num_negs=1,
                 net_dropout=0,
                 aggregator="mean",
                 gamma=0.5,
                 attention_dropout=0,
                 batch_norm=False,
                 net_regularizer=None,
                 embedding_regularizer=None,
                 similarity_score="dot",
                 **kwargs):
        super(SimpleX, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      num_negs=num_negs,
                                      embedding_initializer=embedding_initializer,
                                      **kwargs)
        self.similarity_score = similarity_score
        self.embedding_dim = embedding_dim
        self.user_id_field = user_id_field
        self.user_history_field = user_history_field
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        self.behavior_aggregation = BehaviorAggregator(embedding_dim, 
                                                       gamma=gamma,
                                                       aggregator=aggregator, 
                                                       dropout_rate=attention_dropout)
        self.enable_bias = enable_bias
        if self.enable_bias:
            self.user_bias = EmbeddingLayer(feature_map, 1,
                                            disable_sharing_pretrain=True, 
                                            required_feature_columns=[user_id_field])
            self.item_bias = EmbeddingLayer(feature_map, 1, 
                                            disable_sharing_pretrain=True, 
                                            required_feature_columns=[item_id_field])
            self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(net_dropout)
        self.compile(lr=learning_rate, **kwargs)
            
    def forward(self, inputs):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        user_vecs = self.user_tower(user_dict)
        user_vecs = self.dropout(user_vecs)
        item_vecs = self.item_tower(item_dict)
        y_pred = torch.bmm(item_vecs.view(user_vecs.size(0), self.num_negs + 1, -1), 
                           user_vecs.unsqueeze(-1)).squeeze(-1)
        if self.enable_bias: # user_bias and global_bias only influence training, but not inference for ranking
            y_pred += self.user_bias(self.to_device(user_dict)) + self.global_bias
        loss = self.get_total_loss(y_pred, labels)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs):
        user_inputs = self.to_device(inputs)
        user_emb_dict = self.embedding_layer(user_inputs, feature_source="user")
        user_id_emb = user_emb_dict[self.user_id_field]
        user_history_emb = user_emb_dict[self.user_history_field]
        user_vec = self.behavior_aggregation(user_id_emb, user_history_emb)
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)
        if self.enable_bias: 
            user_vec = torch.cat([user_vec, torch.ones(user_vec.size(0), 1).to(self.device)], dim=-1)
        return user_vec

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec_dict = self.embedding_layer(item_inputs, feature_source="item")
        item_vec = self.embedding_layer.dict2tensor(item_vec_dict)
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        if self.enable_bias:
            item_vec = torch.cat([item_vec, self.item_bias(item_inputs)], dim=-1)
        return item_vec


class BehaviorAggregator(nn.Module):
    def __init__(self, embedding_dim, gamma=0.5, aggregator="mean", dropout_rate=0.):
        super(BehaviorAggregator, self).__init__()
        self.aggregator = aggregator
        self.gamma = gamma
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ["cross_attention", "self_attention"]:
            self.W_k = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                     nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
            if self.aggregator == "self_attention":
                self.W_q = nn.Parameter(torch.Tensor(embedding_dim, 1))
                nn.init.xavier_normal_(self.W_q)

    def forward(self, id_emb, sequence_emb):
        out = id_emb
        if self.aggregator == "mean":
            out = self.average_pooling(sequence_emb)
        elif self.aggregator == "cross_attention":
            out = self.cross_attention(id_emb, sequence_emb)
        elif self.aggregator == "self_attention":
            out = self.self_attention(sequence_emb)
        return self.gamma * id_emb + (1 - self.gamma) * out

    def cross_attention(self, id_emb, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, id_emb.unsqueeze(-1)).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def self_attention(self, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.matmul(key, self.W_q).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def average_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12)
        return self.W_v(mean)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-12)