# =========================================================================
# Copyright (C) 2020-2023. The ReMI Authors. All rights reserved.
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


from torch import nn
import torch
from matchbox.pytorch.models import BaseModel
from matchbox.pytorch.layers import EmbeddingDictLayer, EmbeddingLayer
import torch.nn.functional as F
import numpy as np


class ReMI(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="ReMI",
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)", 
                 embedding_dim=10, 
                 user_id_field="user_id",
                 item_id_field="item_id",
                 user_history_field="user_history",
                 num_negs=1,
                 net_dropout=0,
                 net_regularizer=None,
                 embedding_regularizer=None,
                 similarity_score="dot",
                 interest_num=4,
                 beta=0,
                 reg_ratio=0.1,
                 **kwargs):
        super(ReMI, self).__init__(feature_map,
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
        self.item_id_field = item_id_field
        self.user_history_field = user_history_field
        self.embedding_layer = EmbeddingDictLayer(feature_map, embedding_dim)
        self.dropout = nn.Dropout(net_dropout)
        self.reg_ratio = reg_ratio
        self.max_len = self.feature_map.feature_specs[self.user_history_field]['max_len']
        self.comi_aggregation = ComiRecAggregator(embedding_dim, interest_num=interest_num, seq_len=self.max_len, beta=beta)

        self.compile(lr=learning_rate, **kwargs)
            
    def forward(self, inputs):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        label_ids = item_dict[self.item_id_field].view(labels.size(0), self.num_negs + 1)[:,0]
        user_vecs, readout, atten = self.user_tower(user_dict, {self.item_id_field: label_ids})
        item_vecs = self.item_tower(item_dict)
        if self._stop_training:
            y_pred = torch.bmm(item_vecs.view(user_vecs.size(0), self.num_negs + 1, -1),
                           user_vecs.transpose(1, 2)).max(-1)[0]
        else:
            y_pred = torch.bmm(item_vecs.view(user_vecs.size(0), self.num_negs + 1, -1),
                           readout.unsqueeze(-1)).squeeze(-1)
        loss = self.get_total_loss(y_pred, labels)
        loss += self.reg_ratio * self.calculate_dia_loss(atten)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs, labels):
        user_inputs = self.to_device(inputs)
        user_emb_dict = self.embedding_layer(user_inputs, feature_source="user")
        label_emb_dict = self.embedding_layer(labels, feature_source="item")
        user_history_emb = user_emb_dict[self.user_history_field]
        label_emb = label_emb_dict[self.item_id_field]
        mask = user_history_emb.sum(dim=-1) != 0
        user_vec, read_out, atten, selection = self.comi_aggregation(user_history_emb, label_emb, mask)

        if self.similarity_score == "cosine":
            user_vec = F.normalize(user_vec)
            read_out = F.normalize(read_out)
        return user_vec, read_out, atten

    def item_tower(self, inputs):
        item_inputs = self.to_device(inputs)
        item_vec_dict = self.embedding_layer(item_inputs, feature_source="item")
        item_vec = self.embedding_layer.dict2tensor(item_vec_dict)
        if self.similarity_score == "cosine":
            item_vec = F.normalize(item_vec)
        return item_vec

    def calculate_dia_loss(self, attention):
        C_mean = torch.mean(attention, dim=2, keepdim=True)
        C_reg = (attention - C_mean)
        C_reg = torch.bmm(C_reg, C_reg.transpose(1, 2)) / self.embedding_dim
        dr = torch.diagonal(C_reg, dim1=-2, dim2=-1)
        n2 = torch.norm(dr, dim=(1)) ** 2
        return n2.sum()


class ComiRecAggregator(nn.Module):

    def __init__(self, hidden_size, interest_num=4, seq_len=50, beta=0):
        super(ComiRecAggregator, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_heads = interest_num
        self.interest_num = interest_num
        self.hard_readout = True
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
            nn.Tanh()
        )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
        # self.reset_parameters()

    def forward(self, item_eb, label_eb, mask):
        # item_eb = self.embeddings(item_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))

        # 历史物品嵌入序列，shape=(batch_size, maxlen, embedding_dim)
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w = torch.transpose(item_att_w, 2, 1).contiguous()

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1)  # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)  # softmax之后无限接近于0

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)  # 矩阵A，shape=(batch_size, num_heads, maxlen)

        # interest_emb即论文中的Vu
        interest_emb = torch.matmul(item_att_w,  # shape=(batch_size, num_heads, maxlen)
                                    item_eb  # shape=(batch_size, maxlen, embedding_dim)
                                    )  # shape=(batch_size, num_heads, embedding_dim)

        # 用户多兴趣向量
        user_eb = interest_emb  # shape=(batch_size, num_heads, embedding_dim)

        readout, selection = self.read_out(user_eb, label_eb)
        # scores = None if self.is_sampler or self.name == 'UMI' else self.calculate_score(readout)

        return user_eb, readout, item_att_w, selection

    def read_out(self, user_eb, label_eb):

        # 这个模型训练过程中label是可见的，此处的item_eb就是label物品的嵌入
        atten = torch.matmul(user_eb,  # shape=(batch_size, interest_num, hidden_size)
                             torch.reshape(label_eb, (-1, self.hidden_size, 1))  # shape=(batch_size, hidden_size, 1)
                             )  # shape=(batch_size, interest_num, 1)

        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1),
                          dim=-1)  # shape=(batch_size, interest_num)

        if self.hard_readout:  # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
            readout = torch.reshape(user_eb, (-1, self.hidden_size))[
                (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0],
                                                            device=user_eb.device) * self.interest_num).long()]
        else:  # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)),
                                   # shape=(batch_size, 1, interest_num)
                                   user_eb  # shape=(batch_size, interest_num, hidden_size)
                                   )  # shape=(batch_size, 1, hidden_size)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_size))  # shape=(batch_size, hidden_size)
        # readout是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）
        selection = torch.argmax(atten, dim=-1)
        return readout, selection
