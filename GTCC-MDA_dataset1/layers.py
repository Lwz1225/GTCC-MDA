import math
import torch
from torch import nn
import torch.nn.functional as F
from math import log
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
try:
    from graphormer_dataset1.utils import glorot
except ModuleNotFoundError:
    from utils import glorot





def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)  # 添加一个 Dropout 层，丢弃概率为 0.5，自己加的

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)  # 自己加的
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5  # 根号d分子一  \frac{1}{\sqrt{d}}

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)  实现公式（3）
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)  # 调整为新形状，以便在多头注意力机制中进行并行计算
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # 第一维度和第二维度进行转置       # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # 第一维度和第二维度进行转置       # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # 先转置第一、第二维度，再转置第二、第三维度 # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V  # 实现公式（4）
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]  # 张量 x 进行转置操作，并通过 contiguous() 方法确保结果是一个连续的张量
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size  # 断言语句，用于在程序执行过程中检查一个条件是否满足
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)  # 层归一化
        y = self.self_attention(y, y, y, attn_bias)  # 自注意力模块
        y = self.self_attention_dropout(y)
        x = x + y  # 实现公式（8）

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y  # 实现公式（9）
        return x





class LegacyInnerProductDecoder(nn.Module):
    """
    decoder 解码器
    """
    def __init__(self, output_node_dim, dropout, num_meta):
        super().__init__()
        self.output_node_dim = output_node_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.num_meta = num_meta
        self.weight = nn.Parameter(torch.empty(size=(self.output_node_dim, self.output_node_dim)))  # 建立一个w权重，用于对特征数进行线性变化
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)  # 对权重矩阵进行初始化


    def forward(self, inputs):
        inputs = self.dropout_layer(inputs)
        Meta = inputs[:self.num_meta, :]
        Drug = inputs[self.num_meta:, :]

        Meta = torch.mm(Meta, self.weight)
        Drug = torch.t(Drug)
        x = torch.mm(Meta, Drug)
        # x = torch.reshape(x, [-1])  # 转化为行向量
        outputs = torch.sigmoid(x)
        return outputs


class PairMLPDecoder(nn.Module):
    """MLP classifier over drug-metabolite node-pair features."""

    def __init__(self, embedding_dim, hidden_dim, dropout):
        super().__init__()
        pair_feature_dim = 4 * embedding_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_feature_dim),
            nn.Linear(pair_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings, pair_edge_index, num_meta):
        if pair_edge_index.ndim != 2 or pair_edge_index.size(0) != 2:
            raise ValueError(
                "pair_edge_index must have shape [2, num_pairs]")

        pair_edge_index = pair_edge_index.to(
            device=node_embeddings.device, dtype=torch.long)
        meta_index = pair_edge_index[0]
        drug_index = pair_edge_index[1]
        num_drug = node_embeddings.size(0) - num_meta
        if num_meta <= 0 or num_drug <= 0:
            raise ValueError("node_embeddings must contain metabolite then drug nodes")
        if ((meta_index < 0).any() or (meta_index >= num_meta).any()):
            raise IndexError("metabolite pair indices are out of range")
        if ((drug_index < 0).any() or (drug_index >= num_drug).any()):
            raise IndexError("drug pair indices are out of range")

        # Node order is [metabolite, drug]. Association-matrix indices are
        # [metabolite row, drug column].
        meta_embeddings = node_embeddings[meta_index]
        drug_embeddings = node_embeddings[num_meta + drug_index]

        pair_features = torch.cat(
            [
                meta_embeddings,
                drug_embeddings,
                meta_embeddings * drug_embeddings,
                torch.abs(meta_embeddings - drug_embeddings),
            ],
            dim=1,
        )
        # Return logits; BCEWithLogitsLoss applies sigmoid internally.
        return self.classifier(pair_features).squeeze(-1)
