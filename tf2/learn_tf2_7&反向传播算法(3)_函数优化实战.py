# python3.6
# Create date: 2020-05-19
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &7 反向传播
#   - 7.8 Himmelblau函数优化实战


# ========================

# =======================================================================================================
#                                           第七章   反向传播
# =======================================================================================================
# 7.8 Himmelblau函数优化实战
# ---------------------------------------------------
"""
Himmelblau
f(x, y) = (x**2 +y - 11)**2 + (x + y**2 - 7 )**2

看图会有4个局部极小值点
"""
def himmelblau(x_in):
    x = x_in[0]
    y = x_in[1]
    return  (x ** 2 + y - 11)**2 + (x + y**2 - 7 )**2


x, y = np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
z = himmelblau([X, Y])
# 绘图
import matplotlib.pyplot as plt 
fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y ,z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

## 利用TensorFlow自动求导求出x, y的梯度，并循环迭代更新
for x_ in [[4.0, 0.0], [1.0, 0.0], [-4.0, 0.0], [-2.0, 2.0]]:
    x = tf.constant(x_, dtype=tf.float32)
    print('=='*30)
    print(f'初始化x的值为： {x.numpy()}')
    for step in range(200):
        with tf.GradientTape() as tape:
            tape.watch([x])
            y = himmelblau(x)
        
        grads = tape.gradient(y, [x])[0]
        # 更新参数， 
        lr = 0.01
        x -= lr * grads
        if step % 20 == 19:
            print(f'step {step} : x = {x.numpy()}, f(x) = {y.numpy()} ')


# 7.9 方向传播算法实战
# ---------------------------------------------------
"""
利用前面的多层全连接层的梯度推导结果，直接利用Python循环计算每一层的梯度，
并按梯度下降算法手动更新。采用numpy实现，激活函数为sigmoid
"""
"""
实现一个4层全连接网络实现二分类， 网络输入节点数为2， 隐藏层的节点是为 25， 59， 25
输出层 2， 分别表示类别1的概率和类别2的概率。

输出层直接利用均方差函数计算与One-hot的真实编码之间的误差
"""
## 7.9.1 数据集
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

n_sample = 2000
test_rate = 0.3 
x, y = make_moons(n_sample, noise=0.2, random_state=100)

x_tr, x_te, y_tr, y_te = train_test_split(x, y ,test_size = test_rate, random_state=42)
def make_plot(x, y , title=''):
    fig, axes = plt.subplots(figsize=(10,6))
    axes.scatter(x[y==1, 0], x[y==1, 1], label = 'y=1', c='darkred', alpha=0.7, s = 8)
    axes.scatter(x[y==0, 0], x[y==0, 1], label = 'y=0', c='steelblue', alpha=0.7, s = 8)
    plt.legend()
    plt.title(title)
    plt.show()

make_plot(x, y )


# np.random.normal(0, 0.1, 2 * 50).reshape(2, 50)
# np.random.randn(2, 50) * np.sqrt(1/2)
## 7.9.2 网络层
class Layer():
    def __init__(self, n_input, n_out, activation=None, weights=None
                ,bias=None):
        """
        :param int n_input: 输入节点数
        :param int n_out: 输出节点数
        :param str activation: 激活函数类型
        :param weights: 权重张量， 默认内部生成
        :param bias: 偏置张量，默认内部生成
        """
        # 通过正态分布获取网络权重
        self.weights = weights if weights is not None else np.random.randn(n_input, n_out) * np.sqrt(1 / n_out)
        self.bias = bias if bias is not None else np.random.randn(n_out) * 0.1
        self.activation = activation # 激活函数类型，如sigmoid
        self.last_activation = None # 激活函数的输出值O
        self.error = None # 用于计算当前层的delta变量的中间变量
        self.delta = None # 记录当前层的delta变量， 用于计算梯度

    def activate(self, x):
        # 前向传播
        r = np.dot(x, self.weights) + self.bias # X@W + b
        # 通过激活函数，得到全连接层的输出O
        self.last_activation = self._apply_activation(r)
        return self.last_activation 
    
    def _apply_activation(self, r):
        # 计算激活函数的输出
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1/(1 + np.exp(-r))
        # 其他的时候也默认 直接输出
        return r

    def apply_activation_derivative(self, act_r):
        # 计算激活函数的导数
        # 无激活函数，导数为1
        if self.activation is None:
            return np.ones_like(act_r)
        elif self.activation == 'relu':
            return (act_r > 0) * 1
        elif self.activation == 'tanh':
            return 1 - act_r ** 2
        elif self.activation == 'sigmoid':
            return act_r * (1 - act_r)
        # 其他的时候也默认 直接输出
        return act_r


## 7.9.3 网络模型
class NeuralNetwork():
    def __init__(self):
        self._layers = []
    
    def add_layer(self, layer):
        self._layers.append(layer)
    
    def feed_forward(self, x):
        # 前向传播
        # 就是 sigma(sigma(x@w1 + b1)@w2 + b2)@w3 + b3
        for layer in self._layers:
            x = layer.activate(x)
        return x 

    def backpropagation(self, x, y, learning_rate):
        # 反向传播算法实现
        ## 从后向前计算梯度 
        output = self.feed_forward(x) # 最后层输出
        layer_len = len(self._layers)
        for i in reversed(range(layer_len)):
            layer = self._layers[i] 
            # 如果是输出层
            if layer == self._layers[-1]:
                layer.error = y - output # L = 0.5* (o - y)**2  d(L) = o - y #所以该项式负向的目前 
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                """
                  # ... I --wij -->J --wjk --> K
                  d(L)/d(wij) = d(L)/d(O_k)* d(O_k)/d(O_j) * d(o_j)/d(wij)
                   = ( (y - o) *            # sigma(k)_p1
                       O_k(1-O_k)*wjk *     # error_j_p2    error_j = error_j_p2 * sigma(k)_p1
                       O_j(1-O_j) *         # d(o_j)/d(wij)
                       O_i                  # 上一层输入
                  sigma(j) = error_j * d(o_j)/d(wij)
                """
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # 更新权重
        for i in range(layer_len):
            layer = self._layers[i]
            # o_i为上一层网络输出 保证数据维度在2维及以上
            o_i = np.atleast_2d(x if i == 0 else self._layers[i - 1].last_activation)
            # 梯度下降 因为 d(L)/d(k)  = o -y ，而脚本中是 y - o
            layer.weights += layer.delta * o_i.T * learning_rate

    def train(self, x_train, x_test, y_train, y_test, learning_rate, max_epochs):
        # 网络训练
        # one-hot
        depth_ = 2
        y_onehot = np.zeros((y_train.shape[0], depth_))
        # 索引赋值
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        # 计算mse并更新参数
        mses, accs = [], []
        for i in range(max_epochs + 1):
            for j in range(len(x_train)): # batch=1
                self.backpropagation(x_train[j], y_train[j], learning_rate)
            if i % 10 == 0:
                # 打印mse
                mse = np.mean(np.square(y_onehot - self.feed_forward(x_train)))
                mses.append(mse)
                print('\n','=='*40)
                print(f"Epoch: # {i}, MSE: {mse:.5f}")
                # 打印准确率
                acc = self.accuracy(y_test.flatten() , self.predict(x_test)) * 100
                accs.append(acc/100)
                print(f'Accuracy: {acc:.2f} %','\n')
        return mses

    def predict(self, x):
        return self.feed_forward(x)

    def accuracy(self, y_true, y_pred):
        y_pred_max = np.argmax(y_pred, axis=1)
        corrects = sum(y_true == y_pred_max)
        return corrects/y_pred_max.shape[0]


n_sample = 2000
test_rate = 0.3 
x, y = make_moons(n_sample, noise=0.2, random_state=100)
x_tr, x_te, y_tr, y_te = train_test_split(x, y ,test_size = test_rate, random_state=42)

nn = NeuralNetwork()
nn.add_layer(Layer(2, 25, 'sigmoid')) # 隐藏层1， 2=>25
nn.add_layer(Layer(25, 50, 'sigmoid')) # 隐藏层2， 25=>50
nn.add_layer(Layer(50, 25, 'sigmoid'))
nn.add_layer(Layer(25, 2))


mses = nn.train(x_tr, x_te, y_tr, y_te, 0.01, 1000)

x_len = list(range(len(mses)))
plt.plot(x_len, mses)
plt.show()
     
            
