# python3.6
# Create date: 2020-06-05
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import numpy as np

# ======== 目录 ==========
# &10 卷积神经网络
#   - 10.11 卷积层变种
# ========================

# =======================================================================================================
#                                           第十章   卷积神经网络
# =======================================================================================================

# 10.11 卷积层变种
# -------------------------------------------------
## 10.11.1 空洞卷积

"""
小卷积核使得网络提取特征时的感受野区域有限，但是增大感受野的区域又会增加
网络的参数量和计算代价，因此需要权衡设计。

空洞卷积(Dilated/Atrous Convolution)的提出较好地解决这个问题，空洞卷积在普通卷
积的感受野上增加一个dilation rate 参数，用于控制感受野区域的采样步长。


较大的dilation rate参数并不利于小五台的检测、语义分割等任务

"""
import tensorflow as tf 
from tensorflow.keras import layers
x = tf.random.normal([1, 7, 7, 1])
layer = layers.Conv2D(1, kernel_size=2, strides=1, dilation_rate=2) # dilation_rate=1为一般卷积
out = layer(x)

## 10.11.2 转置卷积
"""
Transposed Convolution 
有时候也有的资料也称反卷积 Deconvolution， 但是并不妥当。

通过输入之间填充大量的padding 来实现 
输出高宽 > 输入高宽的效果
"""
"""
在h=w， 即高宽相等的情况下
in = [2,2];  k = [3, 3]; p = 0
in = [
[-7, -41],
[-15, -81]
]
-->>
步长s=2, 输入数据点之间均匀插入 𝑠 − 1，个空白数据点
in_ = [
[-7, 0, -41],
[0, 0, 0],
[-15, 0, -81]
]
在 3*3 矩阵周围填充相应 k-p-1 = 3 - 0 -1 = 2 列
再在 7*7的输入张量上，进行3*3卷积核， 步长s' = 1， 填充p=0

o = (i + 2*p - k)/s' + 1 = (7 -3)/1 + 1 = 5

# 在 o+2p-k为s倍数时

alpha * (o + 2p -k ) = s
o = alpha*s - 2p + k

"""
# 创建X矩阵，高宽为5*5
x = tf.range(25) + 1
x = tf.reshape(x, [1, 5, 5, 1])
x = tf.cast(x, dtype=tf.float32)
# 创建固定内容的卷积核矩阵
w = tf.constant([[-1, 2, -3.], [4, -5, 6], [-7, 8, -9]])
# 调整为合法维度的张量
w = tf.expand_dims(w, axis=2)
w = tf.expand_dims(w, axis=3)
# 进行普通卷积
out = tf.nn.conv2d(x,w,strides=2,padding='VALID')
out

# 普通卷积的输出作为转置卷积的输入， 进行转置卷积运算
xx = tf.nn.conv2d_transpose(out, w, strides=2,
padding='VALID', output_shape=[1, 5, 5,1])









