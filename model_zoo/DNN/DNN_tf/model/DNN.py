# =========================================================================
# Copyright (C) 2023. The RecBox Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import tensorflow as tf
from recbox.ranking.models import TFRankingModel
from recbox.core.tensorflow.layers import FeatureEmbedding, MLP_Block


class DNN(TFRankingModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DNN", 
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DNN, self).__init__(feature_map, model_id=model_id, **kwargs)
        input_dict = self.get_inputs()
        feature_emb = FeatureEmbedding(feature_map, embedding_dim,
                                       embedding_regularizer=embedding_regularizer)(input_dict, flatten_emb=True)
        y_pred = MLP_Block(output_dim=1,
                           hidden_units=hidden_units,
                           hidden_activations=hidden_activations,
                           output_activation=None, 
                           dropout_rates=net_dropout, 
                           batch_norm=batch_norm,
                           regularizer=net_regularizer)(feature_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        self.build_model(input_dict, return_dict)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
    
