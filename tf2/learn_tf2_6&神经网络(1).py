# python3.6
# Create date: 2020-05-13
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &6 神经网络
#   - 6.1 感知机
#   - 6.2 全连接层

# ========================

# =======================================================================================================
#                                           第六章   神经网络
# =======================================================================================================
# 6.2 全连接层
# ------------------------------------------------
## 6.2.1 张量方式实现
import tensorflow as tf
x = tf.random.normal([2, 784])
w1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x, w1) + b1
o1 = tf.nn.relu(o1)
o1 

## 6.2.2 层方式实现
from tensorflow.keras import layers
x = tf.random.normal([4, 28*28])

fc = layers.Dense(512, activation = tf.nn.relu) # 输出节点为 512
h1 = fc(x) # 在计算fc(x) 自动获取输入
fc.kernel # w
fc.bias # b
# 在优化参数时， 需要获得网络的所有待优化的参数张量列表， 可以通过类的 trainable_variables 查看
fc.trainable_variables # w b
fc.variables 
"""
对于全连接层，内部张量都参与梯度优化，故variables返回列表与 trainable_variables一样。

利用网络层类对象进行前向计算时， 只需要调用类的__call__方法即可，即写成fc(x)方式，
他会自动调用__call_方法，
call实现了 o(X@W + b)  最后返回全连接层的输出张量
"""

# 6.3 神经网络
# ------------------------------------------------
## 6.3.1 张量方式实现
x = tf.random.normal([2, 784])
w1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
w4 = tf.Variable(tf.random.normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

with tf.GradientTape() as tape:
    h1 = tf.nn.relu(x@w1 + b1)
    h2 = tf.nn.relu(h1@w2 + b2)
    h3 = tf.nn.relu(h2@w3 + b3)
    out =  h3@w4 + b4

"""
在使用 TensorFlow自动求导功能计算梯度时，需要将前向计算过程放在tf.GradientTape()环境中，
从而利用GradientTape对象中的gradient()方法自动求解参数的梯度，并利用optimizers对象更新参数
"""

## 6.3.2 全连接层实现
fc1 = layers.Dense(256, activation = tf.nn.relu)
h1 = fc1(x)

fc2 = layers.Dense(128, activation = tf.nn.relu)
h2 = fc2(h1)

fc3 = layers.Dense(64, activation = tf.nn.relu)
h3 = fc3(h2)

fc4 = layers.Dense(10, activation = None)
out = fc4(h3)

# 可以用Sequential 封装为一个网络类
# 类似pipline
model = layers.Sequential(
    layers.Dense(256, activation = tf.nn.relu),
    layers.Dense(128, activation = tf.nn.relu),
    layers.Dense(64, activation = tf.nn.relu),
    layers.Dense(10, activation = None)
)

out = model(x)

## 6.3.3 优化目标
"""
前向传播的最后一步就是完成误差的计算
L = g( f(x) , y) # L称为网络的误差 Loss

f(x) 为利用theata参数化的神经网络模型
g(x) 为误差函数， 用来描述当前网络的预测值与真实标签y之间的差距，如常用的均方差误差函数

theata* = argmin{theata} g( f(x) , y), x<- D^train

argmin{theata}优化问题一般采用 误差反向传播算法求解网络参数theata的梯度信息，并利用梯度下降算法迭代更新：
theata' = theata - lr * (delta_theata L)

(delta_theata L) theata 对于 Loss的偏导

利用误差反向传播算法进行反向计算的过程也叫做反向传播(backward propagation)
"""
"""
从另一个角度来理解神经网络，它完成的是特征的维度变化的功能，比如4层的MNIST手写数字图片识别的全连接网络层，
从 784 -> 256 -> 128 -> 64  -> 10
不断降维浓缩信息，最终的维度包含了与任务相关的高层特征信息，通过对这些特征进行简单的逻辑判定即可完成特定的任务，
如图片分类

参数量计算：
w1 + b1 + w2 + b2 + w3 + b3 + w4 + b4
784 * 256 + 256 + 256 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10 =  242762

"""

# 6.4 激活函数
# ------------------------------------------------

