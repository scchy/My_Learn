# python3.6
# Create date: 2020-06-13
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &11 循环神经网络
#   - 11.1 序列表示方法
#   - 11.2 循环神经网络
#   - 11.3 梯度传播
#   - 11.4 RNN层使用方法
# ========================

# =======================================================================================================
#                                           第十一章   循环神经网络
# =======================================================================================================



"""
自然节的信号除了具有空间维度之外，还有一个时间维度。具有时间维度的信号非常常见。
比如我们正在阅读的文本，说话时发出的语音信号，随着时间变化的股市参数等

"""

# 11.1 序列表示方法
# ---------------------------------------------------
"""
考虑某件商品A在1-6月之间的价格变化趋势，我们记为一维向量[x1, x2, x3, x4, x5, x6]
它的shape为[6], 如果要表示n件商品在1月到6月之间的价格变化趋势，可以记为2维张量 [n, 6]

# b为序列数量， s为序列长度
[b, s]
但是很多信号并不能直接用一个标量数值表示。 比如每个时间戳产生长度为f的特征
[b, s, f]
"""
"""
onehot embedding存在很多问题。所以会有word vevtor使得语义层面的相关性能够很好地在word vector上面体现出来。
一个衡量表示向量的尺度就是余弦相关度。

"""
## 11.1.1 Embedding 层
"""
在神经网络中，单词的表示向量可以直接通过训练的方式得到，我们把单词的标识层叫做Embedding层。
Embedding层负责把单词编码为某个向量vec, 它接受的是采用数字编码的单词idx，

Embedding层实现起来非常简单，通过一个shape为[Nvocab, f]的查询表table,对任意的单词idx，只需要
查询到对应位置上的向量返回即可：
vec = table[idx]

Embedding层是可训练的， 他可放置在神经网络之前，完成单词到向量的装换，得到的表示向量可以继续通过神经网络
完成后续任务，并计算误差L，采用梯度下降来实现端到端的训练。
"""
import tensorflow as tf
from tensorflow.keras import layers
# layers.Embedding(Nvocab, f)
x = tf.range(10)
x = tf.random.shuffle(x)
net = layers.Embedding(10, 4)
out = net(x)
out
## 11.1.2 预训练的词向量
"""
Embedding层的查询表是随机初始化的，需要从空开始训练。实际上，我们可以使用预训练的Word embeddig模型
来得到单词的表示方式，基于训练模型的词向量相当于迁移整个语音空间的知识，往往能得到更好的性能。
"""
# 从预训练模型中加载词向量表
embed_glove = load_embed('glovr.6B.59d.txt')
# 直接利用预训练的词向量初始化Embedding层


# 11.2 循环神经网络
# ---------------------------------------------------
"""
考虑一个句子
I hate this boring movie.

可以转换为[b, s, f], s为句子长度，f为词向量长度。
[1, 5, 10]

"""
## 11.2.1 全连接层可行吗
"""
- 网络参数态多
- 每个全连接层子网络Wi, bi只能感受当前某个单词向量的输入，并不能感知前面的单词语境
和后面单词的语境信息，导致句子整体语义的丢失，每个子网络只能根据自己的输入来提取高层特征
有如管中窥豹

"""
## 11.2.2 权值共享
"""
想当于使用一个全连接网络来提取所有单词的特征信息。
但任然是将句子拆开来分布理解，无法获取整体的语义信息。
"""
## 11.2.3 语义信息
"""
状态张量h，
ht = sigma(Wxh*xt + Whh * h(t-1) + b)
h0 初始化为0
"""
## 11.2.4 循环神经网络
"""
在每个时间戳， 网络层接受当前时间戳的输入xt 和上一个时间戳的网络状态向量h(t-1)
ht = f_theata(h_t-1, x_t)

ht = sigma(Wxh*xt + Whh * h(t-1) + b)
激活函数更多采用tanh函数，可以选择不使用偏置bias来进一步减少参数量。状态向量ht可以直接用作输出，
即ot = ht，也可以对ht做个线性变换ot = w_ho*h_t 得到每个时间戳上的网络输出ot
"""

# 11.3 梯度传播
# ---------------------------------------------------
"""
d(l)/d(whh) = d(l)/d(ot) * d(ot)/d(ht) * d(ht)/d(hi) * d(hi)/d(whh)
其中 d(l)/d(ot) 可以基于算是函数直接求出， ot = ht的情况下：
d(ot)/d(ht) = 1

d(hi)/d(whh) = d(sigma(Wxh*xt + Whh * h(t-1) + b))/d(whh)
i = 1..t

d(ht)/d(hi) = 连乘 d(hk+1)/d(hk)

d(hk)/d(hi) 包含了连乘运算，这是导致循环神经网络训练困难的根本原因
"""

# 11.4 RNN层使用方法
# ---------------------------------------------------
"""
layers.SimpleRNNCell 可以完成 sigma(Wxh*xt + Whh * h(t-1) + b)
Cell仅仅完成一个时间戳的向前运算，不带Cell的层一般是基于Cell实现的，在内部已经完成了
多个时间戳的循环运算，因此使用起来更为方便快捷。
"""
## 11.4.1 SimpleRNNCell
"""
以输入特征长度f = 4，Cell状态向量特征长度为h=3为例，首先我们新建一个SimpleRNNCell，不需要指定长度s
"""
from tensorflow.keras import layers
import tensorflow as tf
cell = layers.SimpleRNNCell(3)
cell.build(input_shape=(None, 4))
cell.trainable_variables
# kernel 变量Wxh张量， recurrent_kernel Whh张量
"""
通过一个List包裹起来，这么写是为了与LSTM, GRU等RNN变种格式统一
ot, [ht] = Cell(xt, [ht-1])

"""

h0 = [tf.zeros([4, 64])]
# b s f
x = tf.random.normal([4, 80, 100])
xt = x[:, 0, :]
# 构建输入特征f=100, 序列长度s=80, 状态长度为64的Cell
cell = layers.SimpleRNNCell(64)
out, h1 = cell(xt, h0)
print(out.shape, h1[0].shape)
# 在序列长度的维度解开输入，得到xt:[b f]
for  xt in tf.unstack(x, axis=1):
    out, h = cell(xt, h) # 前向计算

out

# 11.4.2 多层SimpleRNNCell网络





