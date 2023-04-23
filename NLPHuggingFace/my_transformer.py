# python3
# Create Date: 2023-04-23
# Author: Scc_hy
# Func: transformer 架构
# ======================================================================================

import torch
from torch import nn
from torch.nn import functional as F
import math
from argparse import Namespace


config = Namespace(
    # mluti attention
    hidden_size = 8 * 100,
    num_attention_heads = 8,
    # FeedForward
    intermediate_size=128,
    hidden_dropout_prob=0.1,
    # embedding
    vocab_size=3000,
    max_position_embeddings=2048,
    # num EncoderLayers
    num_hidden_layers=2,
    num_classes=2,
    
    # decoder
    num_decoder_hidden_layers=2,
    decoder_hidden_dropout_prob=0.1,
    decode_embed_flag=False
)



# attetion
# ----------------------------------
def qkv_attention(Q, K, V, mask=None):
    dim_k = K.size(-1)
    A = Q @ K.transpose(1,2) / math.sqrt(dim_k)
    if mask is not None:
        A = A.masked_fill(mask == 0, float("-inf"))
    A = F.softmax(A, dim=-1)
    return A @ V


class AttentionHead(nn.Module):
    def __init__(self, emb_dim, head_dim):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(emb_dim, head_dim)
        self.k = nn.Linear(emb_dim, head_dim)
        self.v = nn.Linear(emb_dim, head_dim)
    
    def forward(self, hidden_state, mask=None):
        return qkv_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state), mask
        )


class EncDecAttentionHead(nn.Module):
    def __init__(self, emb_dim, head_dim):
        super(EncDecAttentionHead, self).__init__()
        self.q = nn.Linear(emb_dim, head_dim)
        self.k = nn.Linear(emb_dim, head_dim)
        self.v = nn.Linear(emb_dim, head_dim)
    
    def forward(self, hidden_state, encode_hidden_state, mask=None):
        return qkv_attention(
            self.q(hidden_state), self.k(encode_hidden_state), self.v(encode_hidden_state), mask
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, config, attention_module=AttentionHead):
        super(MultiHeadAttention, self).__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [attention_module(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, hidden_state, encode_hidden_state=None, mask=None):
        if self.heads[0].__class__.__name__ == 'EncDecAttentionHead' :
            x = torch.cat([h(hidden_state, encode_hidden_state, mask) for h in self.heads], dim=-1)
            return self.out_linear(x)
        
        x = torch.cat([h(hidden_state, mask) for h in self.heads], dim=-1)
        return self.out_linear(x)


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.l2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(self.gelu(x))
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.multi_attion = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        x = x + self.multi_attion(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()
    
    def forward(self, ipt_ids):
        seq_len = ipt_ids.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        token_emb = self.token_embedding(ipt_ids)
        pos_emb = self.pos_embedding(pos_ids)
        embeddings = token_emb + pos_emb
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.embedding = Embedding(config)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x



class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.clf = nn.Linear(config.hidden_size, config.num_classes)
    
    def forward(self, x):
        x = self.encoder(x)[:, 0, :] # select hidden state of [cls] token
        x = self.dropout(x)
        return self.clf(x)




# Docoder

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()
        self.mask_layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.mask_attention = MultiHeadAttention(config)
        
        self.ec_dc_layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.ec_dc_attention = MultiHeadAttention(config, attention_module=EncDecAttentionHead)
        
        self.ffn_layer_norm3 = nn.LayerNorm(config.hidden_size)
        self.ffn = FeedForward(config)

        self.dropout1 = nn.Dropout(config.decoder_hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.decoder_hidden_dropout_prob)
        self.dropout3 = nn.Dropout(config.decoder_hidden_dropout_prob)
    
    def forward(self, x, encode_out):
        # pre layer normaliztion
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        hidden_s = self.mask_layer_norm1(x)
        x = x + self.dropout1(self.mask_attention(hidden_s, mask))
        # atten2 encoder-decoder
        x = x + self.dropout2( self.ec_dc_attention(self.ec_dc_layer_norm2(x), encode_out, mask) )
        # FFN
        x = x + self.dropout3(self.ffn(self.ffn_layer_norm3(x)))      
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.emb_flag = config.decode_embed_flag
        if self.emb_flag:
            self.embedding = Embedding(config)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_decoder_hidden_layers)]
        )
    
    def forward(self, x, encode_out):
        if self.emb_flag:
            x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, encode_out)
        return x



if __name__ == '__main__':
    config.num_classes = 3
    encoder_clf = TransformerForSequenceClassification(config)
    ipts = torch.Tensor([[10, 123, 41, 1230]]).long()
    clf_out = encoder_clf(ipts).size()
    
    enc_out = encoder_clf.encoder(ipts)
    decoder = TransformerDecoder(config)
    out = []
    pred = enc_out
    for _ in range(2):
        pred = decoder(pred, enc_out)
        out.append(pred)
    
    