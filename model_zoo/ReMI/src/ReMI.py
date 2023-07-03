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
from matchbox.pytorch.layers import EmbeddingDictLayer
import torch.nn.functional as F
import math

BACKOFF_PROB = 1e-10


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
                 nce_ratio=0.1,
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
        self.nce_ratio = nce_ratio
        self.max_len = self.feature_map.feature_specs[self.user_history_field]['max_len']
        self.item_num = self.feature_map.feature_specs[self.user_history_field]['vocab_size']
        self.comi_aggregation = ComiRecAggregator(embedding_dim, interest_num=interest_num, seq_len=self.max_len)
        self.nce_sample_loss = NCELoss(noise=self.build_uniform_noise(self.item_num),
                                   noise_ratio=self.num_negs,
                                   norm_term=0,
                                   reduction='elementwise_mean',
                                   per_word=False,
                                   beta=beta,
                                   device=self.device
                                   )
        self.compile(lr=learning_rate, **kwargs)
            
    def forward(self, inputs):
        """
        Inputs: [user_dict, item_dict, label]
        """
        user_dict, item_dict, labels = inputs[0:3]
        label_ids = item_dict[self.item_id_field].view(labels.size(0), self.num_negs + 1)[:,0]
        label_emb_dict = self.embedding_layer({self.item_id_field: label_ids}, feature_source="item")
        label_emb = label_emb_dict[self.item_id_field]
        user_vecs, readout, atten = self.user_tower(user_dict, label_emb)
        item_vecs = self.item_tower(item_dict)
        if self.training:
            y_pred = torch.bmm(item_vecs.view(user_vecs.size(0), self.num_negs + 1, -1),
                               readout.unsqueeze(-1)).squeeze(-1)
        else:
            y_pred = torch.bmm(item_vecs.view(user_vecs.size(0), self.num_negs + 1, -1),
                           user_vecs.transpose(1, 2)).max(-1)[0]

        loss = self.get_total_loss(y_pred, labels)
        loss += self.reg_ratio * self.attention_reg_loss(atten)
        loss += self.nce_ratio * self.nce_sample_loss(label_ids.unsqueeze(-1), readout, self.embedding_layer.embedding_layers[self.item_id_field].weight)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def user_tower(self, inputs, label_emb):
        user_inputs = self.to_device(inputs)
        user_emb_dict = self.embedding_layer(user_inputs, feature_source="user")
        user_history_emb = user_emb_dict[self.user_history_field]
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

    def attention_reg_loss(self, attention):
        C_mean = torch.mean(attention, dim=2, keepdim=True)
        C_reg = (attention - C_mean)
        C_reg = torch.bmm(C_reg, C_reg.transpose(1, 2)) / self.embedding_dim
        dr = torch.diagonal(C_reg, dim1=-2, dim2=-1)
        n2 = torch.norm(dr, dim=(1)) ** 2
        return n2.sum()

    def build_uniform_noise(self, number):
        freq = torch.Tensor([1.0] * number).to(self.device)
        noise = freq / number
        assert abs(noise.sum() - 1) < 0.001
        return noise


class ComiRecAggregator(nn.Module):

    def __init__(self, hidden_size, interest_num=4, seq_len=50):
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


class NCELoss(nn.Module):
    """Noise Contrastive Estimation
    NCE is to eliminate the computational cost of softmax
    normalization.
    There are 3 loss modes in this NCELoss module:
        - nce: enable the NCE approximation
        - sampled: enabled sampled softmax approximation
        - full: use the original cross entropy as default loss
    They can be switched by directly setting `nce.loss_type = 'nce'`.
    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf
    Attributes:
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper), can be heuristically
        determined by the number of classes, plz refer to the code.
        reduction: reduce methods, same with pytorch's loss framework, 'none',
        'elementwise_mean' and 'sum' are supported.
        loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
        are supported
    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: a scalar loss by default, :math:`(B, N)` if `reduction='none'`
    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module
    Return:
        loss: if `reduction='sum' or 'elementwise_mean'` the scalar NCELoss ready for backward,
        else the loss matrix for every individual targets.
    """

    def __init__(self,
                 noise,
                 noise_ratio=100,
                 norm_term='auto',
                 reduction='elementwise_mean',
                 per_word=False,
                 loss_type='nce',
                 beta=0,
                 device=None
                 ):
        super(NCELoss, self).__init__()
        self.device = device
        # Re-norm the given noise frequency list and compensate words with
        # extremely low prob for numeric stability
        self.update_noise(noise)

        self.noise_ratio = noise_ratio
        self.beta = beta
        if norm_term == 'auto':
            self.norm_term = math.log(noise.numel())
        else:
            self.norm_term = norm_term
        self.reduction = reduction
        self.per_word = per_word
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type

    def update_noise(self, noise):
        probs = noise / noise.sum()
        probs = probs.clamp(min=BACKOFF_PROB)
        renormed_probs = probs / probs.sum()
        # import pdb; pdb.set_trace()
        self.register_buffer('logprob_noise', renormed_probs.log())
        self.alias = AliasMultinomial(renormed_probs)

    def forward(self, target, input, embs, interests=None, loss_fn=None, *args, **kwargs):
        """compute the loss with output and the desired target
        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
        """

        batch = target.size(0)
        max_len = target.size(1)
        if self.loss_type != 'full':

            # use all or sampled
            # noise_samples = self.get_noise(batch, max_len)
            noise_samples = torch.arange(embs.size(0)).to(self.device).unsqueeze(0).unsqueeze(0).repeat(batch, 1,
                                                                                                        1) if self.noise_ratio == 1 else self.get_noise(
                batch, max_len)

            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

            # B,N,Nr

            # (B,N), (B,N,Nr)
            if interests != None:
                logit_target_in_model, logit_noise_in_model = self._get_logit_umi(target, noise_samples, input, embs,
                                                                                  interests, *args, **kwargs)
                # noise_batch = embs[noise_samples[0,0].view(-1)]  # Nr X H
                #
                # scores_all = torch.matmul(interests,  # shape=(batch_size, interest_num, hidden_size)
                #                           noise_batch.transpose(1,0))  # shape=( hidden_size,N))
                # value, indice = scores_all.max(dim=-2, keepdim=True)  # (bz,nterest_num, K)
                # scores = torch.gather(scores_all, -2, indice).squeeze()
                #
                # return loss_fn(scores, target)

                # logit_target_in_model, logit_noise_in_model = self._get_logit_umi(target, noise_samples, input, embs, interests, *args, **kwargs)
            else:
                logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
                logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

                logit_target_in_model, logit_noise_in_model = self._get_logit(target, noise_samples, input, embs, *args,
                                                                              **kwargs)

            if self.loss_type == 'nce':
                if self.training:
                    loss = self.nce_loss(
                        logit_target_in_model, logit_noise_in_model,
                        logit_noise_in_noise, logit_target_in_noise,
                    )
                else:
                    # directly output the approximated posterior
                    loss = - logit_target_in_model
            elif self.loss_type == 'sampled':
                loss = self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
            # NOTE: The mix mode is still under investigation
            elif self.loss_type == 'mix' and self.training:
                loss = 0.5 * self.nce_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
                loss += 0.5 * self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )

            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError(
                    'loss type {} not implemented at {}'.format(
                        self.loss_type, current_stage
                    )
                )

        else:
            # Fallback into conventional cross entropy
            loss = self.ce_loss(target, *args, **kwargs)

        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""

        noise_size = (batch_size, max_len, self.noise_ratio)
        if self.per_word:
            noise_samples = self.alias.draw(*noise_size)
        else:
            noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(*noise_size)

        noise_samples = noise_samples.contiguous()
        return noise_samples

    def _get_logit(self, target_idx, noise_idx, input, embs, *args, **kwargs):
        """Get the logits of NCE estimated probability for target and noise
        Both NCE and sampled softmax Loss are unchanged when the probabilities are scaled
        evenly, here we subtract the maximum value as in softmax, for numeric stability.
        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        target_logit, noise_logit = self.get_score(target_idx, noise_idx, input, embs, *args, **kwargs)

        # import pdb; pdb.set_trace()
        target_logit = target_logit.sub(self.norm_term)
        noise_logit = noise_logit.sub(self.norm_term)
        # import pdb; pdb.set_trace()
        return target_logit, noise_logit

    def _get_logit_umi(self, target_idx, noise_idx, input, embs, interests, *args, **kwargs):
        original_size = target_idx.size()
        batch = target_idx.size(0)

        # flatten the following matrix
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)
        sample_size = noise_idx.size(0)
        target_batch = embs[target_idx]
        noise_batch = embs[noise_idx]  # Nr X H

        idx = interests.matmul(target_batch.unsqueeze(-1)).argmax(dim=1)
        select_input = interests[torch.arange(batch), idx.squeeze()]
        target_logit = torch.sum(select_input * target_batch, dim=1)  # N X E * N X E

        idxn = noise_batch.unsqueeze(0).repeat(batch, 1, 1).bmm(interests.transpose(2, 1)).argmax(-1)
        interest_batch = interests[torch.arange(batch).unsqueeze(1), idxn]
        noise_logit = (interest_batch * noise_batch).sum(-1)

        target_logit = target_logit.sub(self.norm_term).unsqueeze(1)
        noise_logit = noise_logit.sub(self.norm_term).unsqueeze(1)
        return target_logit, noise_logit

    def get_score(self, target_idx, noise_idx, input, embs, *args, **kwargs):
        """Get the target and noise score
        Usually logits are used as score.
        This method should be override by inherit classes
        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)
        # import pdb; pdb.set_trace()
        target_batch = embs[target_idx]
        # import pdb; pdb.set_trace()
        # target_bias = self.bias.index_select(0, target_idx)  # N
        target_score = torch.sum(input * target_batch, dim=1)  # N X E * N X E

        noise_batch = embs[noise_idx]  # Nr X H
        noise_score = torch.matmul(
            input, noise_batch.t()
        )
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    def ce_loss(self, target_idx, *args, **kwargs):
        """Get the conventional CrossEntropyLoss
        The returned loss should be of the same size of `target`
        Args:
            - target_idx: batched target index
            - args, kwargs: any arbitrary input if needed by sub-class
        Returns:
            - loss: the estimated loss for each target
        """
        raise NotImplementedError()

    def nce_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the classification loss given all four probabilities
        Args:
            - logit_target_in_model: logit of target words given by the model (RNN)
            - logit_noise_in_model: logit of noise words given by the model
            - logit_noise_in_noise: logit of noise words given by the noise distribution
            - logit_target_in_noise: logit of target words given by the noise distribution
        Returns:
            - loss: a mis-classification loss for every single case
        """

        # NOTE: prob <= 1 is not guaranteed
        logit_model = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        logit_noise = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        # predicted probability of the word comes from true data distribution
        # The posterior can be computed as following
        # p_true = logit_model.exp() / (logit_model.exp() + self.noise_ratio * logit_noise.exp())
        # For numeric stability we compute the logits of true label and
        # directly use bce_with_logits.
        # Ref https://pytorch.org/docs/stable/nn.html?highlight=bce#torch.nn.BCEWithLogitsLoss
        logit_true = logit_model - logit_noise - math.log(self.noise_ratio)

        label = torch.zeros_like(logit_model)
        label[:, :, 0] = 1

        loss = self.bce_with_logits(logit_true, label).sum(dim=2)
        return loss

    def sampled_softmax_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise,
                             logit_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        ori_logits = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        q_logits = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        # subtract Q for correction of biased sampling
        logits = ori_logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()

        if self.beta == 0:
            loss = self.ce(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ).view_as(labels)

        if self.beta != 0:
            x = ori_logits.view(-1, ori_logits.size(-1))
            x = x - torch.max(x, dim=-1)[0].unsqueeze(-1)
            pos = torch.exp(x[:, 0])
            neg = torch.exp(x[:, 1:])
            imp = (self.beta * x[:, 1:] - torch.max(self.beta * x[:, 1:], dim=-1)[0].unsqueeze(-1)).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            if torch.isinf(reweight_neg).any() or torch.isnan(reweight_neg).any():
                import pdb;
                pdb.set_trace()
            Ng = reweight_neg

            stable_logsoftmax = -(x[:, 0] - torch.log(pos + Ng))
            loss = torch.unsqueeze(stable_logsoftmax, 1)

        return loss

class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling
    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.
    Attributes:
        - probs: the probability density of desired multinomial distribution
    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        # @todo calculate divergence
        assert abs(probs.sum().item() - 1) < 1e-5, 'The noise distribution must sum to 1'

        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial
        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)
