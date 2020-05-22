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
#   - 4.9 数学运算
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

"""
轻量级张量复制的手段，在逻辑上扩展张量数据的形状，但只需要在需要时才会执行实际存储复制操作。
对于大部分场景，Broadcasting机制都能通过优化手段避免实际复制数据而完成逻辑运算，从而相对与
tf.tile函数，减少了大量计算代价。


"""
x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])
b = tf.random.normal([3])

# 下属逻辑实际上是
y = x@w + b 

y = x@w + tf.broadcast_to(b, [2, 3]) 
"""
也就是说，操作+ 在遇到shape不一致的2个张量时，会自动考虑将2个张量Broadcasting到一致的shape,
然后再调用 tf.add 完成张量相加运算。
通过自动调用tf.broadcast_to(b, [2,3])的Broadcasting机制，即实现了增加维度、复制数据的目的，又避免实际复制数据的昂贵
计算代价， 同时书写更加简洁高效
"""
"""
Broadcasting机制的核心思想是普适性，即同一份数据普遍适合于其他位置。在验证普适性之前，需要将张量shape靠右对齐，然后进行
普适性判断：
    对于长度为1的维度，默认这个数据普遍适合于当前维度的其他位置。
    对于不存在的维度，则在增加新维度后默认当前数据也普适于新维度，从而可以扩展为更多维度数、其他长度的张量形状。

对于 shape [w, 1] --扩展>> shape [b h w c] 
b h w c 
    w 1  -->> 插入新维度 -->> 1 1 w 1 -->> 扩展为相同长度 -->> b h w c

"""

## 4.9 数学运算
#-----------------------------------------------
### 4.9.1 加减乘除 +-*/
a = tf.range(5)
b = tf.constant(2) 
# 整除 
a//b
# 求余
a%b

### 4.9.2 乘方 tf.pow
a = tf.range(5, dtype=tf.float32)
a ** 2
tf.pow(a, 2)
tf.square(a)
tf.sqrt(a[1:])
a[1:] ** 0.5

### 4.9.3 指数 对数
x = tf.constant([1.0, 2.0, 3.0])
tf.exp(1.)
## 任意底数
tf.math.log(x) / tf.math.log(2.0)


### 4.9.4 矩阵相乘
## tf.matmul(a, b) <=> a@b
"""
tf 的矩阵相乘可以进行批量形式
"""
import tensorflow as tf
a = tf.reshape(tf.range(24), [2,3,4])
b = tf.reshape(tf.range(0, 24), [2,4,3])

a@b
# Broadcasting 
tf.matmul(a, b)




