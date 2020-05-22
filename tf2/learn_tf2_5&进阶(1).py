# python3.6
# Create date: 2020-05-10
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &5 tf进阶
#   - 5.1 合并与分割
#   - 5.2 数据统计
#   - 5.3 张量比较
#   - 5.4 填充与复制
#   - 5.5 数据限幅
#   - 5.6 高级操作

# ========================

# =======================================================================================================
#                                           第五章   TensorFlow进阶
# =======================================================================================================
# 5.1 合并与分割
# ------------------------------------------------
## 5.1.1 合并
"""
设张量A保存了某学校 1-4 号班级的成绩册， 每个班级35个学生，共8门科目
则张量A的shape为 [4, 35, 8]

张量B保存了剩下的6个班级的成绩 [6, 35, 8]
合并2个成绩册，便可得到全学校的张量C [10, 35, 8]

可以使用拼接(Concatenate)和堆叠(Stack)操作实现，拼接并不会产生
新的维度，而堆叠会创建新维度
"""

a = tf.random.normal([4, 35, 8])
b = tf.random.normal([6, 35, 8])
c = tf.concat([a, b], axis=0)
c.shape

### 堆叠
a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
c = tf.stack([a, b], axis=0) # 2科目
d = tf.stack([a, b], axis=-1) # 2班级
c.shape, d.shape

## 5.1.2 分割
"""
全校的成绩 为  [10, 35, 8]
需要切割到全部班级上
tf.split(
    x
    ,axis 分割的维度索引号
    ,num_or_size_splits:
    切割方案
        当为10时，表示切割为10分
        当为[2, 4, 2, 2], 表示切割为4份， 每份的长度为2， 4， 2， 2
)

"""
x = tf.random.normal([10, 35, 8])
res = tf.split(x, axis=0, num_or_size_splits=10)
len(res)

res = tf.split(x, axis=0, num_or_size_splits=[2, 4, 2, 2])
len(res)


# 5.2 数据统计
# ------------------------------------------------
## 5.2.1 向量范数
"""
L1 x所有绝对值之和
L2 所有元素的平方和的开方
oo -范数，定义为向量x的所有元素绝对值的最大值
对于矩阵、张量，同样可以利用向量范数的计算公式，等价于将矩阵、张量打平成向量后计算
"""
x = tf.ones([2, 2])
tf.norm(x, ord=1)
tf.norm(x, ord=2)
tf.norm(x, ord=np.inf)

## 5.2.2 最大最小值、均值、和
"""
tf.reduce_max, tf.reduce_min, tf.reduce_mean, tf.reduce_sum

"""
x = tf.reshape(tf.range(10), [2 , 5])
tf.reduce_max(x, axis=0)
tf.reduce_min(x, axis=0)
tf.reduce_mean(x, axis=0)
tf.reduce_sum(x, axis=0)
tf.reduce_sum(x, axis=1)

# 5.3 张量比较
# ------------------------------------------------
a = tf.random.uniform([100], dtype=tf.int64, maxval=10)
b = tf.random.uniform([100], dtype=tf.int64, maxval=10)

out = tf.equal(a, b)
correct = tf.reduce_sum(tf.cast(out, dtype=tf.float32)).numpy()
acc = correct/out.shape[0]
"""
其他比较函数
tf.math.greater a>b
tf.math.less a<b
tf.math.greater_equal a>=b
tf.math.less_equal a<= b
tf.math.not_equal a!=b
tf.math.is_nan a = nan
"""


# 5.4 填充与复制
# ------------------------------------------------
## 5.4.1 填充
"""
对于 图片数据的高和宽，序列信号的长度，维度长度可能各不相同。
为了方便网络的并行计算，需要将不同的数据扩张为相同长度，之前介绍过复制的方式可以
增加数据的长度，但是重复复制会破坏原有的数据结构，并不适用于此。
通常的做法是，在需要填补长度想信号开始或结束处填充足够数量的特定数值，如0， 使填充后的长度满足
系统要求。那么这种操作叫做填充(Padding)

如：
"I like the weather today." -> [1, 2, 3, 4, 5, 6]
"So do I" -> [7, 8, 1, 6] -->Paddingg--> [7, 8, 1, 6, 0, 0]
"""
import tensorflow as tf 
a = tf.range(1,7)
b = tf.constant([7, 8, 1, 6])
b = tf.pad(b, [[0,2]]) # 填充
tf.stack([a, b], axis=0)

tf.pad(b, [[2,2]]) # 填充 [2,2] list定位维度，2,2定位前后

"""
在自然语言处理中，需要加载不同句子长度的数据集，有些句子长度较小，如10个单词左右，
部分句子长度较长，如超过100个单词。为了能够保存在同一张量中，一般会选取能够覆盖大部分句子长度
的阈值，如80个单词：
    对于小于80个单词的句子，在末尾填充相应数量的0；
    对于大于80个单词的句子，截断超过规定长度的部分单词。
"""
total_words = 10000
max_review_len = 80 # 最大句子长度
embedding_len = 100 # 词向量的长度
# 0- 加载数据
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.imdb.load_data(num_words = total_words)
# 1- 将句子填充或者截断长度，设置末尾填充和末尾截断方式
x_tr = tf.keras.preprocessing.sequence.pad_sequences(x_tr, maxlen=max_review_len, truncating='post', padding='post')
x_te = tf.keras.preprocessing.sequence.pad_sequences(x_te, maxlen=max_review_len, truncating='post', padding='post')
x_tr.shape, x_te.shape

# 前后填充
x = tf.random.normal([4, 28, 28, 1])
tf.pad(x, [[0,0], [2,2,], [2, 2,], [0, 0]]).shape

## 5.4.2 复制
##tf.tile()

# 5.5 数据限幅
# ------------------------------------------------
"""
考虑怎么实现非线性激活函数ReLU的问题。它其实可以通过简答的数据限幅运算实现，
限制数据的范围 x <- [0, +oo)

tf.maximum(x, a) => x <- [a, +oo)
tf.minimum(x, a) => x <- (-oo, a]
"""

def relu(x):
    return tf.maximum(x, 0)

x = tf.range(-3, 9)
relu(x)

# [2, 7]
tf.minimum(tf.maximum(x, 2), 7)


# 5.6 高级操作
# ------------------------------------------------
## 5.6.1 tf.gather
### 类似切片，但是对于不规则的索引， 比切片方便
"""
tf.gather 可以实现根据索引号收集数据的目的。考虑班级成绩册的例子，
共有4个班级， 每个班级35个学生，8门科目 --> [4, 35 ,8]
需要收集第1-2个班级的成绩册，可以给定需要收集班级的索引号：[0, 1]
班级维度为0
"""
a = tf.random.normal([4,35,8])
tf.gather(a, [0,1], axis=0) 
tf.gather(a, [0,3,8,11,12,26], axis=1)


## 5.6.2 tf.gather_nd
"""
希望抽查第2个班级的第2个同学的所有科目，
第3个班级的第3个同学的所有科目，第4个班级的第4个同学的所有科目。
"""
tf.gather_nd(a, [[1,1], [2,2], [3,3]])
# 加科目
tf.gather_nd(a, [[1,1,2], [2,2,3], [3,3,4]])

## 5.6.3 tf.boolean_mask
"""
通过mask的方式采样。
"""
# 采样 1 4班级 
mask = [True, False, False, True]
tf.boolean_mask(a, mask, axis =0).shape

x = tf.random.uniform([2, 3, 8], maxval=100, dtype=tf.int32)
tf.gather_nd(x, [[0, 0], [0,1], [1, 1], [1,2]]) 
tf.boolean_mask(x, [[True, True, False], [False, True, True]])


## 5.6.4 tf.where
a = tf.ones([3,3])
b = tf.zeros([3, 3])

cond = tf.constant([[True, False, False], [False, True, True], [True, True, False]])
tf.where(cond, a, b) #True a  False b
tf.where(cond)

### 例子：获取所有正数
x = tf.random.normal([3, 3])
x

mask = x > 0
# 0- 用boolean_mask
tf.boolean_mask(x, mask)
# 1- 用gather_nd
indices = tf.where(mask)
tf.gather_nd(x, indices)


## 5.6.5 scatter_nd
"""
tf.scatter_nd(indices, updates, shape)可以高效的刷新张量的部分数据， 但是只能在全0张量的白板上刷新，
因此可能需要结合其他操作来实现现有张量的数据刷新功能。
白板的形状为 shape
需要刷新的数据索引为indices
新数据为 updates
返回更新后的白板张量
"""
indices = tf.constant([[4], [3], [1],[7]])
updates = tf.constant([4.4, 3.3, 1.1, 7.7])
tf.scatter_nd(indices, updates, [8])

# 3D 
indices = tf.constant([[1], [3]])
updates = tf.constant([
    [[5]*4, [6]*4, [7]*4, [8]*4],
    [[1]*4, [2]*4, [3]*4, [4]*4]
])
tf.scatter_nd(indices, updates, [4, 4, 4])


## 5.6.6 meshgrid
"""
通过tf.meshgrid可以方便地生成二维网格采样点坐标。
考虑2个自变量x, y的Sinc函数表达式

z = (sin(x^2 + y^2))/(x^2 +y ^2)
"""
x = tf.linspace(-8., 8, 100)
y = tf.linspace(-8., 8, 100)
x, y = tf.meshgrid(x, y)
x.shape, y.shape

z = tf.sqrt(x**2 + y**2)
z = tf.sin(z)/z

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, axes = plt.subplots(figsize = (16, 8))
ax = Axes3D(fig)
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
plt.show()

# 5.7 经典数据加载
# ------------------------------------------------
