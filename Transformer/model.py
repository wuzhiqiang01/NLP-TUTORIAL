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

    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        n_batches = query.size(0)

        query, key, value = [
            lin(x).view(n_batches, -1, self.heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(n_batches, -1, self.heads * self.d_k)
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
    
    def __init__(self, heads, d_model, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
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
    def __init__(self, heads, d_model, d_ff, dropout, N):
        super().__init__()

        self.layers = clones(EncoderLayer(heads, d_model, d_ff, dropout), N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.src_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.dropout1 = nn.Dropout(p=dropout) 

        self.norm2 = LayerNorm(d_model)
        self.cross_atten = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        x_norm = self.norm1(x)
        x_attn = self.src_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout1(x_attn)

        x_norm = self.norm2(x)
        cross_attn = self.cross_atten(x_norm, memory, memory, src_mask)
        x = x + self.dropout2(cross_attn)
        
        x_ffn = self.ffn(x)
        x = x + self.dropout3(x_ffn)
        return x


class Decoder(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout, N):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(heads, d_model, d_ff, dropout), N)
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Embeddings(nn.Module):
    def __init__(self, vcoab_size, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vcoab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(1000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, d_ff, N, heads, dropout) -> None:
        super().__init__()
        self.src_embed = Embeddings(src_vocab, d_model)
        self.pos_embed = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(heads, d_model, d_ff, dropout, N)

    def forward(self, src, tgt, src_mask, tgt_mask):
        x = self.encode(src, src_mask)
        return x

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.pos_embed(x)
        x = self.encoder(x, mask=src_mask)
        return x


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

    transformer = Transformer(1000, 1000, 768, 2048, 6, 8, dropout=0.1)
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    x = transformer(src, src_mask, None, None)
    print(x.shape)


    decoderlayer = DecoderLayer(8, 768, d_ff=2048)
    x = decoderlayer(query, key, None, None)
    print(x.shape)
