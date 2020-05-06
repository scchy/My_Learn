# python3.6
# Create date: 2020-04-20
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &4 tf基础
#   - 4.7 维度变换
#   - 4.8 Broadcasting
# ========================

# =======================================================================================================
#                                           第二章   TensorFlow基础
# =======================================================================================================

## 4.7 维度变换
#-----------------------------------------------
### 4.7.1 reshape
x = tf.range(96)
x = tf.reshape(x, [2, 4, 4, 3])
x
# 按初始视图为[b, h, w, c]
## [b,h*w, c] 张量理解为 b张图片， h*w个像素点，c个通道
## [b,h, w*c] 张量理解为 b张图片， h行，每行的特征长度为 w*c
## [b,h*w*c] 张量理解为 b张图片， 每张图片的特征长度为 h*w*c
x.ndim, x.shape


x = tf.reshape(x, [2, -1, 3])
# -1表示当前轴上程度需要根据视图总元素不变的发展自动推导
#即 
2 * 4* 4 *3/2/3
x.ndim, x.shape

### 4.7.2 增删维度
x = tf.random.uniform([28, 28], maxval=10, dtype=tf.int32)
x.shape
## 1) 增加维度
x1 = tf.expand_dims(x, axis=2)
x1.shape
x2 = tf.expand_dims(x, axis=0)
x2.shape

## 2) 删除维度
x21 = tf.squeeze(x2, axis=0)
## 不指定维度就是将删除所有维度是1的维度
x = tf.random.uniform([1, 28, 28, 1], maxval=10, dtype=tf.int32)
xs = tf.squeeze(x)
"""
>>> x.shape, xs.shape
(TensorShape([1, 28, 28, 1]), TensorShape([28, 28]))
"""

### 4.7.3 交换维度
x = tf.random.normal([2, 32, 32, 3])
x1 = tf.transpose(x, perm=[0,3,1,2])
x.shape, x1.shape

### 4.7.4 数据复制
"""
考虑Y = X@W + b的例子，偏置b插入新维度后， 需要在新维度上复制bacth size份， 将shape变为与 X@W 一致后，才能完成相加

以输入[2, 4]为例，输出为三个节点线性变换层为例，偏置b定义为
b = [1, 2, 3]
"""
b = tf.constant([1, 2, 3])
b = tf.expand_dims(b, axis=0)
B = tf.tile(b, multiples=[2, 1]) # axis=0复制一次
B

x = tf.range(4)
x = tf.reshape(x, [2,2])
tf.tile(x, multiples=[1,3]) # axis=1 复制2次
"""
>>> tf.tile(x, multiples=[1,3])
<tf.Tensor: id=68, shape=(2, 6), dtype=int32, numpy=
array([[0, 1, 0, 1, 0, 1],
       [2, 3, 2, 3, 2, 3]])>
"""

## 4.8 Broadcasting
#-----------------------------------------------


