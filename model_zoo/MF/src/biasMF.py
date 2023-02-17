from torch import nn
import torch
import torch.nn.functional as F
from matchbox.pytorch.models import BaseModel
from matchbox.pytorch.layers import EmbeddingLayer


class BiasMF(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="BiasMF",
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 user_id_field="user_id",
                 item_id_field="item_id",
                 enable_bias=False,
                 num_negs=1,
                 regularizer=None,
                 embedding_dropout=0,
                 similarity_score="dot",
                 **kwargs):
        super(BiasMF, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=regularizer,
                                     num_negs=num_negs,
                                     embedding_initializer=embedding_initializer,
                                     **kwargs)
        self.similarity_score = similarity_score
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        assert similarity_score in ["dot", "cosine", "sigmoid"]
        self.enable_bias = enable_bias
        if self.enable_bias:
            self.user_bias = EmbeddingLayer(feature_map, 1,
                                            disable_sharing_pretrain=True, 
                                            required_feature_columns=[user_id_field])
            self.item_bias = EmbeddingLayer(feature_map, 1, 
                                            disable_sharing_pretrain=True, 
                                            required_feature_columns=[item_id_field])
            self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(embedding_dropout)
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
        if self.enable_bias:
            # user_bias and global_bias only influence training, but not inference phase
            y_pred += self.user_bias(self.to_device(user_dict)) + self.global_bias
        if self.similarity_score == "sigmoid":
            y_pred = y_pred.sigmoid()
        loss = self.get_total_loss(y_pred, labels)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs):
        user_inputs = self.to_device(inputs)
        user_vec = self.embedding_layer(user_inputs, feature_source="user")
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)
        if self.enable_bias: 
            user_vec = torch.cat([user_vec, torch.ones(user_vec.size(0), 1).to(self.device)], dim=-1)
        return user_vec

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec = self.embedding_layer(item_inputs, feature_source="item")
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        if self.enable_bias:
            item_vec = torch.cat([item_vec, self.item_bias(item_inputs)], dim=-1)
        return item_vec

