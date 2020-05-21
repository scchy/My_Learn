
# python3.6
# Create date: 2020-05-21
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &8 Keras高层接口
#   - 8.1 Himmelblau函数优化实战


# ========================

# =======================================================================================================
#                                           第八章   Keras高层接口
# =======================================================================================================


# 8.1 常见功能模块
# ---------------------------------------------------
## 8.1.1 常见网络层
"""
tf.keras.layers命名空间中提供大量常见网络层的类接口，
如全连接层，激活层，池化层，卷积层，循环神经网络层等等

对于这些网络层类，只需要在创建时指定网络层的相关参数，并调用__call__方法即可完成
前向计算。在调用__call__方法时， keras会自动调用每个层的前向传播逻辑，这些逻辑
一般实现在类的call函数中。
"""
"""
tf.nn.softmax 直接在前向输出中进行激活运算。
也可以 layers.Softmax(aix) 搭建网络层 其 axis 参数指定进行softmax运算的维度

"""
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
x = tf.constant([2, 1, 0.1], dtype=tf.float32)
layer = layers.Softmax(axis=-1) # 创建softmax层
layer(x) # 调用 softmax前向计算
# np.exp(2)/ sum([ np.exp(i) for  i in [2, 1, 0.1] ])


## 8.1.2 网络容器
"""
通过 Sequential 封装成一个大网络模型
"""
from tensorflow.keras import layers, Sequential 
network = Sequential([
    layers.Dense(3, activation=None),
    layers.ReLU(),
    layers.Dense(2, activation=None),
    layers.ReLU() 
])
x = tf.random.normal([4, 3])
network(x)
# 也可以通过追加的方法增加网络
layers_num = 2
network = Sequential([])
for _ in range(layers_num):
    network.add(layers.Dense(3))
    network.add(layers.ReLU())

network.build(input_shape=(None, 4))
# layer1 4 * 3 + 3   layer2   3*3 + 3
network.summary()

# 打印网络的待优化参数名与shape
for p in network.trainable_variables:
    print(p.name, p.shape)



# 8.2 模型装配、训练与测试
# ---------------------------------------------------
"""
在训练网络时，一般的流程是通过前向计算获得网络的输出值，再通过损失函数计算网络误差
然后通过自动求导工具计算梯度并更新，同时间隔性的测试网络的性能。
"""

## 8.2.1 模型装配
### 以 Sequential容器封装的网络为例 用于手写识别数字图片识别
## 1- 搭建网络
network = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])
network.build(input_shape = (None, 28*28))

## 2- 优化器 损失函数
from tensorflow.keras import optimizers, losses
### 采用Adam优化器，学习率为0.01； 采用交叉熵损失函数，包含Softmax
network.complie(
    optimizer = optimizers.Adam(lr = 0.01),
    loss = losses.CategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy'] # 测量指标为准确率
)

## 8.2.2 模型训练

