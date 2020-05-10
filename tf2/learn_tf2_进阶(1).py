# python3.6
# Create date: 2020-05-10
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &5 tf进阶
#   - 4.7 维度变换
#   - 4.8 Broadcasting
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



