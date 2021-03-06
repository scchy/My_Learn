# python3.6
# Create date: 2020-05-18
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &7 反向传播
#   - 7.5 全连接层梯度
#   - 7.6 链式法则
#   - 7.7 反向传播算法

# ========================

# =======================================================================================================
#                                           第七章   反向传播
# =======================================================================================================
# 7.5 全连接层梯度
## 实际使用中的神经网络的结构多种多样，我们将以全连接层，激活韩式采用Sigmoid函数，误差函数为Softmax + MSE
## 为例，推导梯度传播方式
# ---------------------------------------------------
## 7.5.1 单个神经元梯度
"""
o^1 = sigma( w^1 * x+ b^1)

输入节点数为j， -->> z^1_1 --sigmoid-->> o^1_1
误差L, t为真实标签
L = 1/2(o^1_1 = t)^2

以权重连接的第j号节点的权重wj1为例， 求偏导
d(L)/d(wj1) = (o1 - t)d(o1)/d(wj1)
将 o1 = sigma(z1) 分解
= (o1 - t)sigma(z1)(1-sigma(z1))d(z1)/d(wj1)
= (o1 - t)o1(1 - o1) *xj

"""
## 7.5.2 全连接层梯度
"""
输入层通过一个全连接层得到输出向量o1,与真实标签向量t计算均方差。
输入节点数为J，输出节点数为K。

同单个神经元不同的是，其多了很多的输出节点。
L = 1/2 *sum([(o^1_i - t_i)**2 for i in range(k)])

由于 d(L) / d(wjk) 仅仅与节点上的 o^1_k有关，上式中的求和符号可以去掉，即i=k:
d(L) / d(wjk) = (ok - tk)ok(1 - ok) *xj

令 delta(k) = (ok - tk)ok(1 - ok)
认为其为变量表征连线的终止节点的梯度传播的某种特征。d(L) / d(wjk) 只与当前连接的起始节点xj，
终止节点处 delta(k)有关。
"""


# 7.6 链式法则
# ---------------------------------------------------
"""
y = f(u)
u = g(x)

d(y)/d(x) = d(y)/d(u) * d(u)/d(x) = f'(u) * g'(x)
= f'(g(x)) * g'(x)

# 多元复合
z = f(x, y), x = g(t), y = h(t)

d(z)/d(t) = d(z)/d(x) * d(x)/d(t) + d(z)/d(y) *d(y)/d(t)

例如：
z = (2t + 1)**2 + e**(t**2) 
令 x = 2t+1, y = t**2 
d(z)/d(t) = 2(2t + 1)*2  + e**(t**2)*2t
=4*(2t+1) + 2te**(t**2)
"""
"""
神经网络的损失函数L来自于输出节点O^K, 其中输出节点O^k又与隐藏层的出输出节点O^J

in -- W_j -->> O_j -- W_k -->> O_k -->> L <--> t

d(L)/d(W_j) = d(L)/d(O_k) * d(O_k)/d(W_j) 

            = d(L)/d(O_k) * d(O_k)/d(O_j) * d(O_j)/d(W_J)
"""
import tensorflow as tf
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)
y = tf.constant(3.)

# 构建梯度记录器：
with tf.GradientTape(persistent=True) as tape:
    # 非tf.Variable类型的张量需要人为设置梯度信息
    tape.watch([w1, b1, w2, b2])
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2
    loss = 0.5 * tf.square(y2 - y)

# 独立求解各个偏导数
dy1_dw1 = tape.gradient(y1, [w1])[0] #  x
dy2_dy1 = tape.gradient(y2, [y1])[0] #  w2
dL_dy2 = tape.gradient(loss, [y2])[0] # (y2 - y)
dL_dw1 = tape.gradient(loss, [w1])[0] # dL_dy2 * dy2_dy1 *  dy1_dw1 = (y2 - y) * w2 * x

print("dL_dw1", dL_dw1)
print("dL_dy2 * dy2_dy1 *  dy1_dw1", dL_dy2 * dy2_dy1 *  dy1_dw1)
print("(y2 - y) * w2 * x", (y2 - y) * w2 * x)



# 7.7 反向传播算法
# ---------------------------------------------------
"""
d(L)/d(wjk) = (ok-t)ok(1-ok)xj = delat_k * xj
考虑网络倒数第二层的偏导数 d(L)/d(WJ) 
第二层输出为O_J, 输出层为O_k ,倒数第三层的输出节点数为 I

int_n --> I -- W_ij --> J -- W_jk --> K --> L <--> T


令 delta_k = (O_k - t_k) * O_k(1 - O_k) * W_jk 

d(L)/d(W_ij) = (O_k - t_k) * d(O_k) /d(W_ij)
           = (O_k - t_k) * O_k(1 - O_k) * d(Z_k)/d(W_ij)
           = (O_k - t_k) * O_k(1 - O_k) * W_jk * d(O_j)/d(W_ij)
           = delta_k * O_j(1 - O_j)  * d(Z_j)/d(W_ij)
           = delta_k * O_j(1 - O_j)  * o_i

令 delta_j = delta_k * O_j(1 - O_j)
d(L)/d(W_ij) = delta_j * o_i
即 写为当前连接的其实节点的输出信息oi 与终止节点j的梯度信息 delta_j的简单相乘的运算

小结下，每层的偏导数的计算公式：
# 输出层
d(L)/d(W_jk) = (O_k - t_k )O_k(1 - O_k) * oj = delta_k * oj
delta_k = O_k(1 - O_k) * (O_k - t_k )

# 倒数第二层
d(L)/d(W_ij) = O_k(1 - O_k) * oj = delta_k*wjk *O_j(1 - O_j)  * o_i = delta_j * o_i
delta_j = delta_k*wjk * O_j(1 - O_j)

# 倒数第三层
d(L)/d(W_ni) = delta_n * O_n 
delta_n = delta_j*wij * O_i(1 - O_i)


"""
