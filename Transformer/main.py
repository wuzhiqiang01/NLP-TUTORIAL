# https://nlp.seas.harvard.edu/annotated-transformer/
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    # stack module deepcopy
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)  # 特征的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_atten = scores.softmax(dim=-1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten, value), p_atten


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        n_batches = query.size(0)

        query, key, value = [
            lin(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(n_batches, -1, self.h * self.d_k)
        )
        x = self.linears[-1](x)
        del key
        del query
        del value
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class EncoderLayer(nn.Module):
    
    def __init__(self, h, d_model, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadedAttention(h, d_model, dropout=dropout)
        self.dropout1 = nn.Dropout(p=dropout)

        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(p=dropout)


    def forward(self, x, mask):
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(x_attn)

        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = x + self.dropout2(x_ffn)
        return x


class Encoder(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout, N):
        super().__init__()

        self.layers = clones(EncoderLayer(h, d_model, d_ff, dropout), N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vcoab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vcoab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Transformer(nn.Module):
    pass


if __name__ == "__main__":
    layer_norm = LayerNorm(features=768)
    x = torch.rand((1, 10, 768))
    print(x.shape)
    y = layer_norm(x)
    print(y.shape)


    query = torch.rand((1, 10, 768))
    key = torch.rand((1, 10, 768))
    value = torch.rand((1, 10, 768))
    attention(query, key, value)

    multiheadattention = MultiHeadedAttention(8, d_model=768)
    x = multiheadattention(query, key, value)
    print(x.shape)


    encoder_layer = EncoderLayer(8, d_model=768, d_ff=2048)
    x = encoder_layer(x, mask=None)
    print(x.shape)


    encoder = Encoder(8, 768, 2048, 0.1, 8)
    x = encoder(x, mask=None)
    print(x.shape)