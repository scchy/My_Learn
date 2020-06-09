# python3.6
# Create date: 2020-06-05
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import numpy as np

# ======== 目录 ==========
# &10 卷积神经网络
#   - 10.11 卷积层变种
#       - 空洞卷积、转置卷积、分离卷积
#   - 10.12 深度残差网络
#   - 10.13 DenseNet
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


"""
o+2p-k 不为s倍数
o=(i+2*p-k)/s + 1
当步长为s>1时，o向下取整
"""
x = tf.random.normal([1, 6, 6, 1])

# 创建固定内容的卷积核矩阵
w = tf.constant([[-1, 2, -3.], [4, -5, 6], [-7, 8, -9]])
# 调整为合法维度的张量
w = tf.expand_dims(w, axis=2)
w = tf.expand_dims(w, axis=3)
# 6*6的输入经过普通卷积核
out = tf.nn.conv2d(x, w, strides=2, padding='VALID')
out.shape 
x = tf.random.normal([1, 6, 6, 1])

"""
a = (o+s*p-k)%s
o=(i-1)*s + k - 2*p + a

tf会自动推导需要填充的行列数a
"""
xx = tf.nn.conv2d_transpose(out, w, strides=2,
padding='VALID',output_shape=[1, 6, 6, 1])
xx

## 矩阵角度
## 转置卷积实现
x = tf.reshape(tf.range(16)+1, [1, 4, 4, 1])
x = tf.cast(x, dtype=tf.float32)
### 创建3*3的卷积核
w = tf.constant([[-1,2,-3],[4,-5,6], [-7, 8, -9]])
w = tf.expand_dims(w, axis=2)
w = tf.expand_dims(w, axis=3)
w

out = tf.nn.conv2d(x, w, strides=1, padding='VALID')
xx = tf.nn.conv2d_transpose(out, w, strides=1, padding='VALID', output_shape=[1,4,4,1])

tf.squeeze(xx)
tf.squeeze(xx)

# 创建转置卷积类
layer_ = tf.keras.layers.Conv2DTranspose(
    1, kernel_size=3, strides=1,padding='VALID'
)
xx2 = layer_(out)
xx2

"""
padding=’VALID’
o=(i-1)*s + k

padding=SAME
o=(i-1)*s + 1
"""

## 10.11.3 分离卷积
"""
普通卷积在对多通道输入进行运算时， 
卷积核的每个通道与输入的每个通道分别进行卷积运算
-->> 得到多通道的特征图
-->> 在对应元素相加产生单个卷积核的最终输出。

分离卷积:
卷积核的每个通道与输入的每个通道分别进行卷积运算
-->> 得到多通道的特征图
-->> 进行多个1*1卷积核运算
-->> 多个高宽不变的输出

"""
"""优势
同样的输入和输出， 采用分离卷积的参数约是普通卷积的1/3

普通卷积：3*3*3*4=108
分离卷积：3*3*3*1+1*1*3*4 = 41


分离卷积在Xception 和 MobileNets等对计算代价敏感的邻域中
得到了大量应用。
"""


# 10.12 深度残差网络
# -------------------------------------------------
"""
网络的层数越深， 越可能获得更好的泛化能力。
但是模型加深以后， 网络变得越来越难训练，这主要是由于梯度弥散现象造成的。

一种很自然的想法是，既然浅层神经网络不容易出现梯度弥散现象，那么可以尝试给深层神经网络添加一种
回退到浅层神经网络的机制。当深层神经网络可以轻松地回退到浅层神经网络时，深层神经网络可以获得与
浅层神经网络相当的模型性能，而不至于更糟糕

"""
## 10.12.1 RestNet原理
# ResNet 通过在卷积层的输入和输出之间添 Skip Connection 实现层数回退机制
"""
H(x) = F(x) + x

一般需要x 与 F(x)的shape 完全一致， 所以一般都会进行 额外的卷积运算

identity(x) 以1*1的卷积运算居多，主要用于调整输入的通道数

"""
## 10.12.2 RestBlock实现
import tensorflow as tf
from tensorflow.keras import layers, Sequential

class BasicBlock(layers.Layer):
    """
    残差模块类
    """
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # f(x)包含2个普通卷积层， 创建卷积层1
        self.conv1 = layers.Conv2D(filter_num, kernel_size=(3,3), strides=stride, paddig='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 创建第二卷积层
        self.conv2 = layers.Conv2D(filter_num, kernel_size=(3,3), strides=stride, paddig='same')
        self.bn2 = layers.BatchNormalization()   

        if stride != 1: # 插入identity层 用于实现 x 与 f(x) shape 一致
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D, kernel_size=(1,1),  strides=stride)
        else: # 否则直接连接
            self.downsample = lambda x:x 
        
    def call(self, inputs, training=None):
        # 前向传播
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 输入通过identity()转换
        identity = self.downsample(inputs)
        # f(x) + x
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return  output



# 10.13 DenseNet
# -------------------------------------------------








