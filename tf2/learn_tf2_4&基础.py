# python3.6
# Create date: 2020-04-20
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &2 回归问题
# &3 分类问题
# &4 tf基础

# ========================

# =======================================================================================================
#                                           第二章   回归问题
# =======================================================================================================

# 2.3 线性模型实战
import numpy as np
# 1- 数据加载
x = np.random.uniform(-10, 10, 100)
eps = np.random.normal(0, 0.1, 100)
y = 1.477 * x + 0.089 + eps
dt = np.c_[x, y]

# 2- 计算误差
def mse(b, w, dt):
    m = dt.shape[0]
    return sum(dt[:,1] - (w * dt[:,0] + b) ** 2) / m

def step_gradient(b, w, dt, lr):
    m = dt.shape[0]
    x, y  = dt[:,0], dt[:,1]
    b_delt = sum(2/m * ((w*x + b) - y))
    w_delt = sum(2/m * x * ((w*x + b) - y))
    return b - lr * b_delt, w -lr * w_delt

def gradient_desc(dt, lr, iters):
    # 初始化b w
    loss = 0
    b, w = np.random.uniform(0, 2, 2)
    for step in range(iters):
        b, w = step_gradient(b, w, dt, lr)
        loss = mse(b, w, dt)
        if step % 50 == 0:
            print(f'iteration: {step}, loss:{round(loss, 5)}, w:{round(w, 3)}, b:{round(b, 3)}')
    return b, w, loss


b, w, loss = gradient_desc(dt, lr=0.01, iters=500)
print('=='*10, '\n', f'loss:{round(loss, 5)}, w:{round(w, 3)}, b:{round(b, 3)}')

"""
>>> b, w, loss = gradient_desc(dt, lr=0.01, iters=500)
iteration: 0, loss:-35.42897, w:1.045, b:1.7
iteration: 50, loss:-68.07179, w:1.485, b:0.667
iteration: 100, loss:-67.69558, w:1.479, b:0.29
iteration: 150, loss:-67.62812, w:1.476, b:0.151
iteration: 200, loss:-67.61289, w:1.475, b:0.1
iteration: 250, loss:-67.60858, w:1.475, b:0.082
iteration: 300, loss:-67.60717, w:1.475, b:0.075
iteration: 350, loss:-67.60668, w:1.475, b:0.072
iteration: 400, loss:-67.6065, w:1.475, b:0.071
iteration: 450, loss:-67.60644, w:1.475, b:0.071

========================================
 loss:-67.60641, w:1.475, b:0.071
"""

# =======================================================================================================
#                                           第三章   分类问题
# =======================================================================================================
# 0- 加载数据
## 张量缩放到 -1， 1
(x, y), (x_val, y_cal ) = datasets.mnist.load_data()
x = 1 * tf.convert_to_tensor(x, dtype=tf.float32)/255 - 1 

# 1- y值处理
y = tf.convert_to_tensor(y, dtype=tf.int32)
y_unique = tf.unique(y).y.shape[0]
y = tf.one_hot(y, depth = y_unique) # on-hot编码
print(x.shape, y.shape)

# 2- 构建训练集
tr_dt = tf.data.Dataset.from_tensor_slices((x, y))
tr_dt = tr_dt.batch(512) # 批量训练

# 3- 构建模型
## 3-1 由于线性模型表达能力偏弱，所以需要转化为非线性
# =======================================================================================================
#                                           第四章   tensorflow基础
# =======================================================================================================


## 4.1 数据类型
#------------------------------------------------
### 4.1.1 数值型
a = 1.2 
aa = tf.constant(a)
type(a), type(aa), tf.is_tensor(aa)

x = tf.constant([1, 2, 3.3])
x.numpy()

# 可以通过list和np.array 赋值到tensor
tf.constant([[1, 2, 3],[2,3,4]])
tf.constant(np.arange(8).reshape((2,2,2)))

### 4.1.2 字符串类型
a = tf.constant('Hello, Deep Learning')
a
tf.strings.lower(a)

tf.strings.lower('Hello, Deep Learning')

### 4.1.3 bool
a = tf.constant([True, False])
print(a)
# 需要注意 类型是 tensor != True
a == True


## 4.2 数值精度
#------------------------------------------------
# 保存精度过低时， 数据123456789发生了溢出， 得到了错误的结果，
# 一般使用 tf.int32, tf.int64精度， 浮点数，一般采用tf.float32
tf.constant(123456789, dtype=tf.int16) 
tf.constant(123456789, dtype=tf.int32)

tf.constant(np.pi, dtype=tf.float32)
tf.constant(np.pi, dtype=tf.float64)
"""
对于大部分深度学习算法，一般使用tf.int32, tf.float32 可满足运算精度要求，部分对精度要求较高的算法，
如强化学习，可以选择使用 tf.int64, tf.float64
"""
### 4.2.1 读取精度 & 类型装换
a = tf.constant(123456789.3, dtype=tf.float16) 
print(f'before: {a.dtype}', a)
if a.dtype != tf.float32:
    a = tf.cast(a, tf.float32)

print(f'after: {a.dtype}', a)

a = tf.constant([True, False])
tf.cast(a, dtype = tf.int32)


## 4.3 待优化张量
#------------------------------------------------
"""
为了区分需要计算梯度信息的张量和不需要计算梯度信息的张量，TensorFlow 增加了一种专门的数据类型来支持梯度信息的记录：
tf.Variable.
tf.Variable 类型在普通的张量类型基础张添加了name,trainable等属性标签来支持计算图的构建。由于梯度运算会消耗大量的计算资源
，而且会自动更新相关参数，对于不需要的优化的张量，如神经网络的输入X,不需要通过tf.Variable封装；相反的，对于需要计算梯度优化的张量，
如神经网络层的W和b, 需要通过tf.Variable包裹一遍TensorFlow跟踪相关梯度信息
"""
a = tf.constant([-1, 0, 1, 2])
aa = tf.Variable(a)
aa.name, aa.trainable

tf.Variable([-1, 0, 1, 2])
# 待优化的张量可看作普通张量的特殊类型，普通张量也可以通过GradientTape.wattch()方法临时加入跟踪梯度信息的列表


## 4.4 张量创建
#------------------------------------------------
### 4.4.1 全1 0张量
tf.zeros([]), tf.ones([])
tf.zeros([1]), tf.ones([1])
tf.zeros([2,2]), tf.ones([3,2])
a = tf.constant([[1,2], [3,4]])
tf.zeros_like(a)

### 4.4.2 创建自定义数值张量
tf.fill([2,2], 9)

### 4.4.3 创建已知分布的张量
tf.random.normal([2, 2])
tf.random.normal([2, 2], mean=10, stddev= 2)
# 均匀分布张量
tf.random.uniform([2, 2], minval=0, maxval=10)
tf.random.uniform([2, 2], minval=0, maxval=100,  dtype=tf.int32)

### 4.4.5 创建序列
tf.range(10, delta=2)

## 4.5 张量的典型应用
#------------------------------------------------
### 4.5.1 标量
"""
维度0， shape为[]
标量的典型用途之一是误差值的表示、各种测量指标的表示，如acc,precision, recall
"""
out = tf.random.uniform([4, 10])
y = tf.constant([2, 3, 2,0])
y = tf.one_hot(y, depth=10)
loss = tf.keras.losses.mse(y, out)
loss = tf.reduce_mean(loss)
print(loss)

### 4.5.2 向量
z = tf.random.normal([4, 2])
b = tf.zeros([2]) # 设置偏置向量
z = z + b         # 添加偏置
"""
通过高层接口类Dense()方式创建的网络层，张量W和b存储在类的内部，由类自动创建并管理。

可以通过全连接层的bias成员比那里查看偏置b，例如创建输入节点数为4，输出节点数为3的线性层网络
，那么它的偏置向量的程度为3：
"""
from tensorflow.keras import layers 
fc = layers.Dense(3) # 创建一层wx+b， 输出节点为3 
# 通过build函数创建w, b张量，输入节点为4
fc.build(input_shape=(2, 4))
fc.bias

### 4.5.3 矩阵
x = tf.random.normal([2, 4])
w = tf.ones([4, 3]) # 因为输出是三所以需要三列的 w 
b = tf.zeros([3])
o = x@w + b
"""
x，w 张量均为矩阵，x@w+b网络层称为线性层，在tf中可以通过Dense类直接实现，
Dense层也称为全连接层。 我们通过Dense类创建输入 4 个节点， 输出3个节点的网络层， 
可以通过全连接层的 kernel成员名查看其权重矩阵W
"""
fc = layers.Dense(3)  # 定义全连接层的输出节点为3 
fc.build(input_shape=(2, 4))
fc.kernel # w 


### 4.5.4 3维张量
"""
典型应用是标识序列信号，格式为：
X = [b, sequence len, feature len]
b标识序列信号的数量
squence len 表示序列信号在时间维度上的采样点数
feature len 标识每个点的特征长度

考虑自然语言处理中句子的表示，如评价句子的是否为正面情绪的情感分类任务网络
为了能够方便字符串被神经网络处理，一般将单词通过嵌入层(embeding layer)编码为固定长度的向量
比如'a' 编码为某个长度3的向量，那么2个等长（单词为5）的句子序列可以表示为
[2, 5, 3]的3维张量
"""
(x_tr, y_tr), (x_te, t_te) = tf.keras.datasets.imdb.load_data(num_words=10000)
x_tr = tf.keras.preprocessing.sequence.pad_sequences(x_tr, maxlen=80)
x_tr.shape
# 可以看到，其中25000表示句子个数，80表示每个句子共80个单词， 每个单词使用数字编码方式。
# 可以通过layers.Embedding层将数字编码的单词换换为程度为100个词向量
embedding = tf.keras.layers.Embedding(10000, 100)
out = embedding(x_tr)
out.shape
# 可以看到，经过Embedding层编码后，句子张量的shape变为[25000, 80, 100]
# 其中100表示每个单词编码为长度100的向量
"""
对于特征长度为1 的序号， 比如商品价格在60天内的变化曲线， 只需要一个标量即可表示商品的价格，
因此2件商品的价格变化趋势可以使用shape为[2, 60], 为了统一格式，将价格变化趋势表达为 [2, 60,1]
1为一个特征长度
"""

### 4.5.5 4维张量
"""
4维张量在卷积神经网络中应用的非常广泛，它统御保存特征图数据一般定义为：
[b, h, w, c]
b是输入的数量，h/w分别便是特征图的高宽，c表示特征图的通道数

"""
# 创建 32*32 的彩色图片输入，个数为4
x = tf.random.normal([4, 32, 32, 3])
# 创建卷积神经网络
layer = tf.keras.layers.Conv2D(16, kernel_size=3)
out = layer(x)
out.shape
layer.kernel.shape

## 4.6 索引&切片
#------------------------------------------------
x = tf.random.normal([4, 32, 32, 3])
x[0][1]
x[0, 1]

x[1:3]
# ... 多个维度全取
x[0:2,...,1:]  # 1-2张图片的 G/B通道数据




