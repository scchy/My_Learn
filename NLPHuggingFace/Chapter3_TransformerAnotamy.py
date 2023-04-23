# python3
# Create Date: 2023-04-23
# Author: Scc_hy
# Func: Transformer Anotamy
# tip: pip install datasets
# =================================================================================

# -----------------------------------------------------
# 一、简介
# -----------------------------------------------------
__doc__ = """
Transformer 分成encoder-decoder

一般有三种应用方式：
1. Encoder-only
    应用: 文本分类、实体识别
    input: 文本序列（s双方向输入 bidirectional attention.)
    转换: 数值代表
    output: 基于任务的 n-dim 
    代表模型: BERT、RoBERTa、DistilBERT
    
2. Decoder-only
    应用：文本生成-AIGC
    input: 文本序列,（单方向输入 causal or autoregressive attention.)
    转换: 数值代表
    output: 最有可能的下一个文本 （文本序列）
    代表模型: GPT
    
3. Encoder-Decoder
    应用： 机器翻译, 文本总结
    input: 文本序列 
    转换: 数值代表
    output: 文本序列
    代表模型: BART、T5

在实际应用中 其应用不是绝对的，单个encode decoder同样可以做encoder-decoder的任务
"""

# -----------------------------------------------------
# 二、self-attetion : Encoder
# -----------------------------------------------------

## 2.1 可视化attention
from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show

model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = "time flies like an arrow"
show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)


## 2.2 分步实现attention
## ---------------------
from torch import nn
from transformers import AutoConfig

### 生成I
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
# torch.Size([batch_size=1, seq_len=5, hidden_dim=768])
inputs_embeds = token_emb(inputs.input_ids)
I = inputs_embeds

### Q = W_q @ I / K = W_k @ I -> A = Q^TK -> scale & softmax -> 
import torch
from math import sqrt
from torch.nn import functional as F
Q = K = I # 简便操作可以直接用自己作为Q K
dim_k = K.size(-1)
# scores = torch.bmm(Q, K.transpose(1,2)) / sqrt(dim_k)
A = Q @ K.transpose(1,2) / sqrt(dim_k)
A.size()
A = F.softmax(A, dim=-1)

### O = A @ V
V = I
O = A @ V # torch.bmm(A, V)
O.size()

def scaled_dot_product_attention(Q, K, V):
    dim_k = K.size(-1)
    A = Q @ K.transpose(1,2) / sqrt(dim_k)
    A = F.softmax(A, dim=-1)
    return A @ V

scaled_dot_product_attention(I, I, I)


## 2.3 分步实现 multi-attention
## ---------------------
class AttentionHead(nn.Module):
    def __init__(self, emb_dim, head_dim):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(emb_dim, head_dim)
        self.k = nn.Linear(emb_dim, head_dim)
        self.v = nn.Linear(emb_dim, head_dim)
    
    def forward(self, hidden_state, mask=None):
        return scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state), mask
        )
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        return self.out_linear(x)
    

### 2.3.1 可视化多头Attention
# -----------------------
from bertviz import head_view
from transformers import AutoModel

model = AutoModel.from_pretrained(model_ckpt, output_attentions=True)
sentence_a = "time flies like an arrow"
sentence_b = "fruit flies like a banana"

viz_ips = tokenizer(sentence_a, sentence_b, return_tensors='pt')
attentions = model(**viz_ips).attentions
sentence_b_start = (viz_ips.token_type_ids == 0).sum(dim=1)
tokens = tokenizer.convert_ids_to_tokens(viz_ips.input_ids[0])

head_view(attentions, tokens, sentence_b_start, heads=[8])




## 2.4 Feed-Forward layer
##  就是二层全连接层
## ---------------------

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


## 2.5 Adding Layer Normalization
##  增加layer标准化
## ---------------------
some_info = """
Post layer normalization
    后归一化，这个在论文中使用，Normalize 放在skip connect之后：
        (start-skip1)-> MultiHeadAttention - (+skip1) -> LayerNorm (start-skip2) -> FF - (+skip2) -> LayerNorm
    这样做让训练有点困难，因为梯度会发散。所以在transformer训练的时候常常会用到warnUp的方法

Pre layer normalization
    前置归一化，这样的结构训练会更加的稳定，并不需要warm-up.将Normalize放在 start-skip后
        (start-skip1)-> LayerNorm -> MultiHeadAttention -(+skip1) ->  (start-skip2) LayerNorm -> FF -(+skip2) ->
"""
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        # pre layer normaliztion
        hidden_s = self.layer_norm1(x)
        x = x + self.attention(hidden_s)
        x = x + self.ffn(self.layer_norm2(x))
        return x

## 2.6 Positional Embedding
## 位置信息, 可以和token一样的处理
## ----------------------------
some_info = """
Absolute Positional representations
    直接进行cos sin然后合并， 这种操作在词汇量不大的时候非常有效

Relative Positional representations
    用Embedding. 在DeBERTa 中用了这个方法
"""

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


## 2.7 组合成TransformerEncoder
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


## 2.8 增加一个分类的head
# ----------------------------------
class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.clf = nn.Linear(config.hidden_size, config.num_class)
    
    def forward(self, x):
        x = self.encoder(x)[:, 0, :] # select hidden state of [cls] token
        x = self.dropout(x)
        return self.clf(x)


config.num_classes = 3
encoder_clf = TransformerForSequenceClassification(config)
ipts = torch.Tensor([[10, 123, 41, 1230]]).long()
encoder_clf(ipts.iput_ids).size()
        

# -----------------------------------------------------
# 三、self-attetion : Decoder
# decoder 和 encoder 唯一的区别就是 有两个 attention sublayers
# -----------------------------------------------------
some_info = """
Masked multi-head self-attention layer
    确保我们在每个时间步生成的tokens仅仅依赖于之前的outputs。
    可以通过tril来实现mask

Encoder-decoder attention layer
    Performs multi-head attention:
        Q: decoder out
        K: encoder out
        V: encoder out
    去提取decoder-out 与 encoder输出词之间的相关性 在对齐翻译（Attention-nmt）中用到

"""

seq_len = 4
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
mask_res = """
tensor([[[1., 0., 0., 0.],
         [1., 1., 0., 0.],
         [1., 1., 1., 0.],
         [1., 1., 1., 1.]]])
"""
# 对输入可以用masked_fill 进行处理
mask.masked_fill(mask == 0, -float('inf'))


def scaled_dot_product_attention(Q, K, V, mask=None):
    dim_k = K.size(-1)
    A = Q @ K.transpose(1,2) / sqrt(dim_k)
    if mask is not None:
        A = A.masked_filled(mask == 0, float("-inf"))
    A = F.softmax(A, dim=-1)
    return A @ V

