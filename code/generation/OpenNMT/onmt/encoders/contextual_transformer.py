import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
from typing import Tuple
import onmt
from onmt.encoders.encoder import EncoderBase
from onmt.modules.position_ffn import PositionwiseFeedForward


class Cross_Transformer(nn.Module):

    def __init__(self, configs, embeddings, embeddings1, alpha=0.5):
        super(Cross_Transformer, self).__init__()
        self.word_length = configs.max_word_length
        self.alpha = alpha
        # self.num_layers = num_layers
        # self.d_model = d_model
        # self.heads = heads
        # self.dropout = dropout
        self.embeddings = embeddings
        self.embeddings1 = embeddings1

        self.contextual_transformer = TextEntity_Transformer(
            configs.contextual_transform, configs.contextual_transform.output_dim, embeddings, embeddings1)

        self.contextual_transformer2 = TextEntity_Transformer(
            configs.contextual_transform, configs.contextual_transform.output_dim, embeddings, embeddings1)


        # self.conv = nn.Conv2d(2048, 768, 1)
        # self.bn = nn.BatchNorm2d(768)


    def forward(self, e, f):
        # e 是问题 f 是实体
        cap_lengths = len(e)
        # print('e', e.size())
        # print('f', f.size())
        # print('f', f)
        
        # e_f_mask = torch.ones(cap_lengths, 16).cuda()
        # f_e_mask = torch.ones(cap_lengths, 16).cuda()

        words = e[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        e_mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len).cuda()

        words_f = f[:, :, 0].transpose(0, 1)
        w_batch_f, w_len_f = words_f.size()
        padding_idx_f = self.embeddings1.word_padding_idx
        # print('padding', padding_idx_f)  # 1
        f_mask = words_f.data.eq(padding_idx_f).unsqueeze(1).expand(w_batch_f, w_len_f, w_len_f).cuda()

        # print(words.data.eq(padding_idx).size())

        e_f_mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len_f, w_len).cuda()
        f_e_mask = words_f.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len_f).cuda()

        # print('f_e_mask', f_e_mask.size())

        e = torch.squeeze(e, dim=1) # [batch_size, 40, 768]
        # e1 = e[:, :self.word_length, :]
        # e2 = e[:, self.word_length: self.word_length*2, :]
        # e3 = e[:, self.word_length*2:, :]
        # e = self.fc(e) # [batch_size, 40, 64]

        f = torch.squeeze(f, dim=1)
        # f = F.relu(self.bn(self.conv(f)))  # [batch_size, 768, 4, 4]
        # f = f.view(f.shape[0], f.shape[1], -1)  # [batch_size, 768, 16]
        # f = f.permute([0, 2, 1])  # [batch_size, 16, 768]

        c_e_f = self.contextual_transformer(e, e_mask, e_f_mask, f)  # e问题 f实体 计算问题的编码

        # print('c_e_f', c_e_f.size())
        # print('-----------------------------')
        c_f_e = self.contextual_transformer2(f, f_mask, f_e_mask, e)  # 计算实体的编码

        # c1_e1_f = self.contextual_transform(e1, e_f_mask, f)
        # c1_f_e1 = self.contextual_transform2(f, f_e_mask, e1)
        a = self.alpha

        # c1 = a * c1_e1_f + (1 - a) * c1_f_e1

        # c2_e2_f = self.contextual_transform(e2, e_f_mask, f)
        # c2_f_e2 = self.contextual_transform2(f, f_e_mask, e2)

        # c2 = a * c2_e2_f + (1 - a) * c2_f_e2

        # c3_e3_f = self.contextual_transform(e3, e_f_mask, f)
        # c3_f_e3 = self.contextual_transform2(f, f_e_mask, e3)

        # c3 = a * c3_e3_f + (1 - a) * c3_f_e3

        # x = torch.cat((c1, c2, c3), dim=1)

        c_e_f = c_e_f.transpose(0, 1).contiguous()
        c_f_e = c_f_e.transpose(0, 1).contiguous()


        return c_e_f, c_f_e



class LayerNormalization(nn.Module):
    def __init__(self, features_count, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(
            torch.ones(features_count), requires_grad=True)
        self.bias = nn.Parameter(
            torch.zeros(features_count), requires_grad=True)
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class TextEntity_Transformer(nn.Module):
    def __init__(self, ct: EasyDict, feature_dim: int, embeddings, embeddings1):
        super().__init__()

        self.input_norm = LayerNormalization(feature_dim)
        input_dim = feature_dim
        self.embedding = embeddings
        self.embedding1 = embeddings1
        
        # self.embedding = PositionalEncoding(
        #     input_dim, ct.dropout, max_len=1000)

        self.tf = TransformerEncoder(
            ct.num_layers, input_dim, ct.num_heads, input_dim,
            ct.dropout)

        self.use_context = ct.use_context
        if self.use_context:
            self.tf_context = TransformerEncoder(
                ct.atn_ct_num_layers, input_dim, ct.atn_ct_num_heads,
                input_dim, ct.dropout)
        
        self.linear = nn.Linear(input_dim * 2, input_dim)
        
        init_network(self, 0.01)

    def forward(self, src, self_mask, cross_mask, src1):
        # features = self.input_norm(features)
        if src.size()[-1] != 768:
            emb = self.embedding(src)
        else:
            emb = src
        out = emb.transpose(0, 1).contiguous()
        
        features = self.tf(out, out, out, self_mask)
        add_after_pool = None

        if src1.size()[-1] != 768:
            src1 = self.embedding1(src1)
        else:
            src1 = src1
        src1 = src1.transpose(0, 1).contiguous()
        # src1 = self.tf(src1, src1, src1, mask)

        if self.use_context:
            ctx = self.tf_context(src1, features, features, cross_mask)
            add_after_pool = ctx    # ctx.squeeze(1)

        # print('features', features.size())
        # pooled = torch.mean(features, dim=1)
        # print('add_after_pool', add_after_pool.size())
        # add_after_pool = torch.mean(add_after_pool, dim=1)
        # if add_after_pool is not None:

        batch_size, src_len, d_model = features.size()
        batch_size, src1_len, d_model = add_after_pool.size()
        
        # add_after_pool = linear(add_after_pool)

        linear_layer = nn.Linear(src1_len, src_len).cuda()
        reshaped_pool = add_after_pool.contiguous().view(-1, src1_len)
        pool = linear_layer(reshaped_pool)
        add_after_pool = pool.view(batch_size, src_len, d_model)
        # print('add_after_pool', add_after_pool.size())

        pooled = torch.cat([features, add_after_pool], dim=-1)
        pooled = self.linear(pooled)
        return pooled

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout_prob=0., max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        dimension = torch.arange(0, dim).float()
        div_term = 10000 ** (2 * dimension / dim)
        pe[:, 0::2] = torch.sin(position / div_term[0::2])
        pe[:, 1::2] = torch.cos(position / div_term[1::2])
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        if step is None:
            x = x + self.pe[:x.size(1), :]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob):
        super().__init__()
        self.d_model = d_model
        assert layers_count > 0
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads_count, d_ff, dropout_prob)
                for _ in range(layers_count)])

    def forward(self, query, key, value, mask):
        # print('query', query.size())
        # print('key', key.size())
        # print('mask', mask.size())
        batch_size, query_len, embed_dim = query.shape
        batch_size, key_len, embed_dim = key.shape
        # mask = (1 - mask.unsqueeze(1).expand(batch_size, query_len, key_len))
        # mask = mask == 1
        sources = None
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(query, key, value, mask)
        return sources


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention_layer = MultiHeadAttention(heads_count, d_model, dropout_prob)
        self.pointwise_feedforward_layer = PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, sources_mask):
        # print('type', type(key))
        sources = self.self_attention_layer(query, key, value, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)
        return sources


# class Sublayer(nn.Module):
#     def __init__(self, sublayer, d_model):
#         super(Sublayer, self).__init__()
#         self.sublayer = sublayer
#         self.layer_normalization = LayerNormalization(d_model)

#     def forward(self, *args):
#         x = args[0]  # 实际上是Q
#         x = self.sublayer(*args) + x
#         return self.layer_normalization(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads_count, d_model, dropout_prob):
        super().__init__()
        assert d_model % heads_count == 0,\
            f"model dim {d_model} not divisible by {heads_count} heads"
        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, d_model = query.size()
        batch_size, key_len, d_model = key.size()
        d_head = d_model // self.heads_count
        # print('query_', query.size())
        # print('key_', key.size())

        '''最开始的地方就修改query的维度'''
        # linear_layer = nn.Linear(query_len, key_len).cuda()
        # reshaped_tensor = query.contiguous().view(-1, query_len)
        # query = linear_layer(reshaped_tensor)
        # query = query.view(batch_size, key_len, d_model)
        
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)
        # print('key', key_projected.size())
        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()
        query_heads = query_projected.view(
            batch_size, query_len, self.heads_count, d_head).transpose(1, 2)
        
        key_heads = key_projected.view(
            batch_size, key_len, self.heads_count, d_head).transpose(1, 2)
        
        value_heads = value_projected.view(
            batch_size, value_len, self.heads_count, d_head).transpose(1, 2)
        
        

        # key_heads_new = torch.zeros(batch_size, self.heads_count, query_len, d_head)
        # value_heads_new = torch.zeros(batch_size, self.heads_count, query_len, d_head)
        # key_heads_new[:, :, :key_len, :] = key_heads
        # value_heads_new[:, :, :value_len, :] = value_heads
        # print('value_heads', value_heads_new.size())
        
        # 修正维度
        '''1'''
        # linear_layer1 = nn.Linear(key_len, query_len).cuda()
        # linear_layer2 = nn.Linear(value_len, query_len).cuda()
        # key_heads = linear_layer1(key_heads)
        # value_heads = linear_layer2(value_heads)
        '''2'''
        # linear_layer = nn.Linear(query_len, key_len).cuda()
        # reshaped_tensor = query_heads.contiguous().view(-1, query_len)
        # map_tensor = linear_layer(reshaped_tensor)
        # query_heads = map_tensor.view(batch_size, self.heads_count, key_len, d_head)

        # linear_layer1 = nn.Linear(key_len, query_len).cuda()
        # reshaped_tensor1 = key_heads.contiguous().view(-1, key_len)
        # map_tensor1 = linear_layer1(reshaped_tensor1)
        # key_heads = map_tensor1.view(batch_size, self.heads_count, query_len, d_head)

        # linear_layer2 = nn.Linear(value_len, query_len).cuda()
        # reshaped_tensor2 = value_heads.contiguous().view(-1, value_len)
        # map_tensor2 = linear_layer2(reshaped_tensor2)
        # value_heads = map_tensor2.view(batch_size, self.heads_count, query_len, d_head)
        # print('value_heads', value_heads.size())
        # print('key_heads', key_heads.size())
        # print('query_heads', query_heads.size())

        attention_weights = self.scaled_dot_product(
            query_heads, key_heads)
        # print('attention_weight', attention_weights.size())
        if mask is not None:
            # print('mask_shape', mask.size())
            # mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            mask_expanded = mask.unsqueeze(1).repeat(1, self.heads_count, 1, 1)
            # print('mask_expand', mask_expanded.size())
            attention_weights.masked_fill(mask_expanded, -1e18)
        attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(attention)
        context_heads = torch.matmul(
            attention_dropped, value_heads)
        # print('context_heads', context_heads.size())  # batch, heads, seq_len, d_model
        context_sequence = context_heads.transpose(1, 2)
        # context = context_sequence.reshape(
        #     batch_size, query_len, d_model)
        context = context_sequence.contiguous().view(batch_size, -1, self.heads_count * d_head)
        final_output = self.final_projection(context)
        final_output = self.layer_normalization(final_output + query)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(
            query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob))
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, x):
        x_ = self.feed_forward(x)
        x = self.layer_normalization(x_ + x)
        return x

def truncated_normal_fill(
        shape: Tuple[int], mean: float = 0, std: float = 1,
        limit: float = 2) -> torch.Tensor:
    num_examples = 8
    tmp = torch.empty(shape + (num_examples,)).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)

def init_weight_(w, init_gain=1):

    w.copy_(truncated_normal_fill(w.shape, std=init_gain))


def init_network(net: nn.Module, init_std: float):

    for key, val in net.named_parameters():
        if "weight" in key or "bias" in key:
            init_weight_(val.data, init_std)
