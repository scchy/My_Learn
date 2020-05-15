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
#   - 6.3 神经网络
#   - 6.4 激活函数
#   - 6.5 输出层设计
#   - 6.6 误差计算
#   - 6.7 神经网络类型

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
## 6.4.1 Sigmod 
"""
sigmoid: = 1/(1+e^-x)
相对于跃阶函数，可以直接利用梯度下降算法优化网络参数，应用的非常广泛
"""
import tensorflow as tf
x = tf.linspace(-6.0, 6.0, 10)
tf.nn.sigmoid(x)

## 6.4.2 ReLU
"""
修正线性单元，
Sigmoid函数在输入值较大或较小时，容易出现梯度值接近0的现象，称为梯度弥散现象，使得网络参数长时间得不到更新，
很难训练较深层次的网络模型。
ReLU:= max(0, x)

2001年，神经科学家Dayan和Abott模拟得出更加精确的脑神经源激活模型，
它具有单侧抑制、相对宽松的兴奋边界等特性，
"""
tf.nn.relu(x)


## 6.4.3 LeakyReLU
"""
ReLU 函数在 x<0时梯度值恒为0， 也可能或造成梯度弥散现象，为了克服这个问题，
LeakyReLU函数被提出，

LeakyReLU={X X>=0; P*X X<0}

"""
tf.nn.leaky_relu(x, alpha =0.1)



## 6.4.4 Tanh
"""
tanh(x) = (e^x - e^-x) / (e^x + e-x)
= 2 * sigmoid(2x) - 1

"""
tf.nn.tanh(x)

# 6.5 输出层设计
# ------------------------------------------------
"""
o <- R^d 输出属于整个实数空间， 或者某段普通的实数空间， 比如函数值趋势的预测
o <- [0, 1] 输出值特别地落地在[0,1]的区间， 如二分类的问题的概率
o <- [0, 1] 输出值落在[0, 1]的区间， 并且所有输出值之和为1， 常见的如多分类问题
o <- [-1, 1]
"""
## 6.5.1  普通实数空间
"""
像正弦函数曲线预测、年的预测、股票走势的预测等
输出层可以不加激活函数。误差的计算直接基于最后一层的
输出o与真实值y进行计算，如采用均方差误差函数度量输出值o与真实值y之间的距离
L = g(o, y)

"""

## 6.5.2 [0,1]区间
"""
如果直接输出，会分布在实数空间，所以需要添加某个合适的激活函数
如Sigmoid,
同样的，对于二分类问题，如硬币的正反面的预测，输出层可以只需要一个节点，
某个事件A发生概率P(A|x)。如果我们把网络的输出o表示正面事件出现的概率，
那么反面事件出现的概率为 1-o，
P(正面|x) = o
p(反面|x) 1-o

"""
## 6.5.3 [0,1]区间
"""
P(A|x) + P(B|x) + P(C|x) = 1
用softmax
o(z_i) = (e^(z_i) ) / sum([e^(z_j) for j in n_out])


"""
def soft_max(x):
      return tf.exp(x)/tf.reduce_sum(tf.exp(x))

x = tf.constant([2.0, 1.0, 0.1])
soft_max(x)

tf.nn.softmax(x)
"""
与Dense层类似，Softmax函数也可以作为网络层使用， 通过 layers.Softmax(axis=-1)
在Softmax函数的数值计算过程中，容易因输入值偏大方式数值溢出现象；在计算交叉熵的时候，
也会出现数值溢出的问题。为了数值计算的稳定性，Tensorflow中提供了一个统一的接口，将Softmax
与交叉熵损失函数同时实现，同时也处理了数值不问他的异常，一般推荐使用，避免单独使用softmax函数和
交叉熵损失函数。函数式接口
tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logists=False)
"""
z = tf.random.normal([2, 10]) # 构造输出层
y_onehot = tf.constant([1,3])
y_onehot = tf.one_hot(y_onehot, depth=10)
# 输出层未使用softmax函数，故from_logits设置为True
loss = tf.keras.losses.categorical_crossentropy(y_onehot, z, from_logits=True)
loss = tf.reduce_mean(loss)

## 6.5.4 [-1, 1]
x = tf.linspace(-6., 6., 10)
tf.tanh(x)


# 6.6 误差计算
# MSE = sum((y-o)^2)/n
# ------------------------------------------------
## 6.6.1 均方差误差
o = tf.random.normal([2, 10])
y = tf.constant([1,3])
y_onehot = tf.one_hot(y, depth=10)
#  tf.reduce_mean(tf.square(y_onehot - o), axis=1)
loss = tf.keras.losses.MSE(y_onehot, o) # 函数返回的是每个样本的均方差
loss = tf.reduce_mean(loss) # 求一个batch的loss
loss
# 或者通过MSE类处理
criteon = yf.keras.losses.MeanSquaredError()
loss = criteon(y_onehot, o)
loss

## 6.6.2 交叉熵
"""
熵越大，代表不确定性越大，信息也就越大。
H(P) = -1*sum([ p_i * np.log2(p_i) for p_i in range(n) ])

对于四分类， y_true = [0, 0, 0, 1]
H(p(y is 4|x)) = -1*sum([ p_i * np.log2(p_i) for p_i in [0, 0, 0, 1] ]) = 0 # 0的时候取0 
即不确定性最低
如果预测时 [0.1, 0.1, 0.1, 0.7]
-1*sum([ p_i * np.log2(p_i) for p_i in [0.1, 0.1, 0.1, 0.7] ]) = 1.356

如果预测概率是均等的情况下
-1*sum([ p_i * np.log2(p_i) for p_i in [0.25, 0.25, 0.25, 0.25] ]) = 2.0
"""
"""
交叉熵(Cross Entropy)
H(p, q) = -sum([p_i * np.log2(q_i) for p_i, q_i in zip(p_lst, q_lst)])
# 可以分解成p 的墒与 p q 的KL散度的和：
H(p, q) = H(p) + D_kl(p|q)
D_kl(p|q) = sum([p_x*np.log2(p_x/q_x) for p_x, q_x in  zip(p_lst, q_lst)])

KL散度是用于衡量2个分布之间距离的指标， p=q时， D_kl(p|q)=0。 交叉熵和KL散度都是不对称的
H(p, q) != H(q, p)
D_kl(p|q)  != D_kl(q|p) 
KL可以很好的衡量2个分布之间的差别，特别的，当分类问题中y的编码分布p采用one-hot 编码时： H(y) = 0
此时：
H(y, o) = H(y) + D_kl(y|o) = D_kl(y|o) 

根据KL散度的定义，可以推导出问题中交叉熵的计算表达式：
H(y, o) = D_kl(y|o) = sum([p_x*np.log2(p_x/q_x) for p_x, q_x in  zip(p_lst, q_lst)])
y -> y_onhot 由于进行了one-hot编码

H(y, o) = D_kl(y|o) = 1*np.log2(1/oi) = -np.log2(oi)
可以看到，L只与真实类别i上的概率oi有关，对应概率oi越大，H(y, o)越小，当对应概率为1时， 交叉熵H(y, o)
取得最小值0，此时网络输出o与真实标签y完全一致，神经网络取得最优状态。

最小化交叉熵的过程也就是最大化政企类别的预测概率的过程。
"""
y = tf.constant([1])
y_onehot = tf.one_hot(y, depth=3)

def H_p(y_):
    """
    计算熵
    """
    # 0- 计算熵
    h_lst = -tf.math.log(y_) / tf.math.log(2.0)
    # 1- 修正-np.inf
    # 1-1- 获取非-np.inf位置
    mask = h_lst.numpy() == np.inf 
    indices = tf.where(mask)
    # 1-2- 更新-np.inf
    if sum(mask[0]):
        update_zero = np.zeros(len(indices))
        h_lst_update = tf.scatter_nd(indices, update_zero, h_lst.shape)
    else:
        h_lst_update = h_lst
    return tf.reduce_sum(h_lst_update)

a = tf.constant([[0.1, 0.7, 0.2]])
H_p(a)
H_p(y_onehot)

def crossentropy(y_, y_p):
    mask = y_.numpy() ==  1.0
    indices = tf.where(mask)
    print(mask, indices)
    ypi = tf.gather_nd(y_p, indices)
    out = - tf.math.log(ypi) / tf.math.log(2.0)
    return tf.constant([0.]) if sum(out.numpy() == np.inf) else out

crossentropy( y_onehot, tf.constant([[0.1, 0.9, 0.2]]) )




# 6.7 神经网络类型
# ------------------------------------------------
"""
全连接层前向计算简单，梯度求导也较简单，但是它有一个最大的缺陷， 在处理较大特征长度的数据时，
全连接层的参数往往较大，使得训练深层数的全连接网络比较困难。

"""
## 6.7.1 卷积神经网络 
"""
分析理解图片
通过利用局部相关性和权值共享的思想

"""
## 6.7.2 循环神经网络
"""
序列信号 文本数据

LSTM网络作为RNN的变形，克服了RNN缺乏长期记忆，不擅长处理长序列的问题，在自然语言处理中得到了广泛的应用。
基于LSTM谷歌做了Seq2Seq模型，并成功商用于谷歌神经网络及其翻译系统(GNMT)


"""
## 6.7.3 注意力(机制)网络 2017 
## 6.7.4 图神经网络


