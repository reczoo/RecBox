from torch import nn
import torch
from matchbox.pytorch.models import BaseModel
from matchbox.pytorch.layers import MLP_Layer, EmbeddingLayer
import torch.nn.functional as F


class YouTubeNet(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="YouTubeNet", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 output_dim=10,
                 user_hidden_units=[],
                 user_hidden_activations="ReLU",
                 user_final_activation=None,
                 num_negs=1,
                 embedding_dropout=0,
                 net_dropout=0,
                 batch_norm=False,
                 net_regularizer=None,
                 embedding_regularizer=None,
                 similarity_score="dot",
                 sample_weighting=False,
                 **kwargs):
        super(YouTubeNet, self).__init__(feature_map, 
                                         model_id=model_id, 
                                         gpu=gpu, 
                                         embedding_regularizer=embedding_regularizer,
                                         net_regularizer=net_regularizer,
                                         num_negs=num_negs,
                                         sample_weighting=sample_weighting,
                                         embedding_initializer=embedding_initializer,
                                         **kwargs)
        self.similarity_score = similarity_score
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        num_user_fields = feature_map.get_num_fields(feature_source="user")
        self.user_dnn_layer = MLP_Layer(input_dim=embedding_dim * num_user_fields,
                                        output_dim=output_dim, 
                                        hidden_units=user_hidden_units,
                                        hidden_activations=user_hidden_activations,
                                        final_activation=user_final_activation, 
                                        dropout_rates=net_dropout, 
                                        batch_norm=batch_norm) \
                              if user_hidden_units is not None else None
        self.dropout = nn.Dropout(embedding_dropout)
        self.compile(lr=learning_rate, **kwargs)
            
    def forward(self, inputs, training=True):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        user_vecs = self.user_tower(user_dict)
        user_vecs = self.dropout(user_vecs)
        item_vecs = self.item_tower(item_dict)
        y_pred = torch.bmm(item_vecs.view(-1, self.num_negs + 1, self.embedding_dim), 
                           user_vecs.unsqueeze(-1)).squeeze(-1)
        return_dict = {"y_pred": y_pred}
        if training:
            loss = self.get_total_loss(y_pred, labels)
            return_dict["loss"] = loss

        return return_dict

    def user_tower(self, inputs):
        user_inputs = self.to_device(inputs)
        user_embedding = self.embedding_layer(user_inputs, feature_source="user")
        user_vec = user_embedding.flatten(start_dim=1)
        if self.user_dnn_layer is not None:
            user_vec = self.user_dnn_layer(user_vec)
        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)
        return user_vec

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec = self.embedding_layer(item_inputs, feature_source="item")
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        return item_vec