# python3.6
# Create date: 2020-05-21
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &8 Keras高层接口
#   - 8.1 常见功能模块
#   - 8.2 模型装配、训练与测试
#   - 8.3 模型保存与加载
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

# 0- 数据加载
from tensorflow.keras import datasets
(x, y), (x_val, y_val ) = datasets.mnist.load_data()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.     #先将类型转化为float32，再归一到0-1
    x = tf.reshape(x, [-1, 28*28])              #不知道x数量，用-1代替，转化为一维784个数据
    y = tf.cast(y, dtype=tf.int32)              #转化为整型32
    y = tf.one_hot(y, depth=10)                 #训练数据所需的one-hot编码
    return x, y


train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(60000)      #尽量与样本空间一样大
train_db = train_db.batch(100)          #128
train_db = train_db.map(preprocess)


test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.shuffle(10000)      #尽量与样本空间一样大
test_db = test_db.batch(100)          #128
test_db = test_db.map(preprocess)

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
network.compile(
    optimizer = optimizers.Adam(lr = 0.01),
    loss = losses.CategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy'] # 测量指标为准确率
)


## 8.2.2 模型训练

"""
epochs是指训练迭代的epochs数， validation_data是指用于验证的数据集
和验证的评论 validation_freq 
    每个 batch 中间隔几次 验证一次

verbose
0 = silent, 1 = progress bar, 2 = one line per epoch
"""
history = network.fit(train_db, epochs=5, validation_data = test_db
                    ,validation_freq = 100, verbose=2)


history.history

## 8.2.3 模型测试
x, y = next(iter(test_db))
out = network.predict(x) # 模型预测
print(out)
## 如果只是简单地测试模型的性能， 可以通过Model.evaluate(db)即可循环完db数
## 据集上所有的样本，并打印性能
network.evaluate(test_db)



# 8.3 模型保存与加载
# ---------------------------------------------------
## 8.3.1 张量方式
"""

网络的状态主要体现在网络的结构以及网络层的内部张量参数上，因此在 【拥有】
【网络结构源】 文件的条件下，直接保存网络张量参数到文件上是最轻量级的一种方式。
"""
network.save_weights('weights.ckpt')
print('saved weights.')

network.load_weights('weights.ckpt') # 直接在已有的网络结构上加载
print('loaded weights!')

## 8.3.2 网络方式
network.save('model.h5')
print('saved total model.')

network = tf.keras.models.load_model('model.h5') # 重构网络+导入参数

## 8.3.3 SavedModel方式
"""
tf有强大的生态系统， 包括移动端和网页端的支持。当需要模型部署到其他平台时，采用tf
剔除的SaveModel方式更具有平台无关性
"""
tf.keras.experimental.export_saved_model(network, 'model-savedmodel') # 载入在文件夹里面
print('export saced model') 

# 用户无需关心文件的保存格式，只需要通过 从文件恢复网络结构与网络参数
network = tf.keras.experimental.load_from_saved_model('model-savedmodel')


