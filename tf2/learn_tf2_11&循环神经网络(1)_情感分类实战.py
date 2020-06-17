    
# python3.6
# Create date: 2020-06-17
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &11 循环神经网络
#   - 11.5 RNN情感分类问题实战
# ========================

# =======================================================================================================
#                                           第十一章   循环神经网络
# =======================================================================================================

# 11.5 RNN情感分类问题实战
# ---------------------------------------------------
"""
利用基础的RNN网络挑战情感分类问题。网络结构如11.9
2层RNN， 循环提取序列信号的语义特征， 利用第二层Rnn的最后时间戳的
状态向量h^2_s作为句子的特征表示，送入全连接层构成的分类网络3， 得到样本x为积极情感的概率P
(x作为积极情感|x) <- [0, 1]
"""
## 11.5.1 数据集
"""
IMDB < 5 用户标记为0 即消极， IMDB > 7 用户标记为1 即积极
25000 条影评用于训练集，25000条用于测试集。
"""
batches = 128
total_words = 1000 # 词汇表大小 n_vocab
max_review_len = 80 # 句子最大程度s, 大于的句子部分将截断， 小于的将填充
embedding_ken = 100 # 词向量特征长度f 
# 加载IMDB数据集，此处的数据采用数字编码，一个数字代表一个单词
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.imdb.load_data(num_words=total_words)
print(x_tr.shape, len(x_tr[10]), y_tr.shape )
print(x_te.shape, len(x_te[10]), y_te.shape )

n = 0
word_index = tf.keras.datasets.imdb.get_word_index()
for k, v in word_index.items():
    print(k, v)
    n += 1
    if n == 10:
        break

# 由于编码表的键为单词，值为ID，我们需要翻转编码表，并添加标志位的编码ID

# 前面4 个ID 是特殊位
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3
# 翻转编码表
reverse_word_index = dict([(value, key) for (key, value) in
word_index.items()])
# 对于数字编码的句子， 通过如下函数装换为字符串数值
def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])

decode_review(x_te[0])

# 构建数据集
def get_db_imdb(x, y, batch_=128):
    db_ = tf.data.Dataset.from_slice((x, y))
    shuffle_num = x.shape[0]
    db_ = db_.shuffle(shuffle_num)
    db_ = db_.batch(batch_, drop_remainder=True) # 最后个bacth不要
    return db_

tr_db = get_db_imdb(x_tr, y_tr, batches)
te_db = get_db_imdb(x_te, y_te, batches)

print('x_train shape:', x_tr.shape, tf.reduce_max(y_tr),
tf.reduce_min(y_tr))
print('x_test shape:', x_te.shape)

## 11.5.2 网络模型
class MyRNN(tf.keras.Model):
    # Cell方式构建多层网络
    def __init__(self, batchsz, units):
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([batchsz, units])]
        self.state0 = [tf.zeros([batchsz, units])]
        # 词向量编码 [b 80] -> [b 80 100]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        # 构建2个cell
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        # 构建分类网络， 用于将CELL的输出特征进行分类
        # [b 80 100] -> [b 64] -> [b 1]
        self.outlayer = layers.Dense(1)
    
    def call(self, inputs, training=None):
        x = inputs # [b, 80]
        # embedding -->> [b, 80, 100]
        x = self.embedding(x)
        # run cell compute [b 80 100] --> [b, 64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            # word [b, 100]
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell0(out0, state1, training)
        # 末层最后一个输出作为分类网络的输入 [b, 64] => [b , 1]
        x = self.outlayer(out1, training)
        # p (y is pos|x)
        prob = tf.sigmoid(x)
        return prob
    

## 11.5.3 训练与测试

