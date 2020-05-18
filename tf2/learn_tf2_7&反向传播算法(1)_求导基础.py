# python3.6
# Create date: 2020-05-16
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &7 反向传播
#   - 7.1 感知机
#   - 7.2 导数
#   - 7.3 激活函数导数
#   - 7.4 损失函数

# ========================

# =======================================================================================================
#                                           第七章   反向传播
# =======================================================================================================
# 7.3 激活函数导数
# ------------------------------------------------
## 7.3.1 Sigmiod 
#### 经过推导 Sigmiod(x)' = Sigmiod(x)(1-Sigmiod(x))

import numpy as np
def sigmiod(x):
    return 1/(1+np.exp(x))

def derivative(x):
    """
    Simiod'
    """
    return sigmiod(x) * (1 - sigmiod(x))

## 7.3.2 Relu 
"""
ReLU' = 1, X>=0; 0, X <0;
因此它不会梯度变的无限大 ，造成梯度爆炸
也不会梯度变的无限小，造成梯度弥散
"""

a = np.array([-1, -1, -.05, 1, 2, 4])
def derivative(x):
    """
    Relu'
    """
    return (x >=0) *1 

derivative(a)


## 7.3.3 LeakyRelu 
"""
LeakyRelu' = 1, X>=0; p, X <0;
"""
def derivative(x, p):
    """
    LeakyRelu'
    """
    out = np.ones_like(x)
    out[x < 0] = p
    return out

derivative(a, 0.01)

## 7.3.4 Tanh函数梯度
"""
tanh(x) = (e^x - e^-x) / (e^x + e^-x)
= 2* sigmiod(2x) - 1

tanh(x) = 1- tanh(x)

# (f + g)' = f' + g'
# (fg)' = f'*g + f *g'
# (f/g)' = (f'g - fg') / (g^2)

"""

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def derivative(x):
    """
    tanh'
    """
    return 1 - tanh(x) ** 2



# 7.4 损失函数
# ------------------------------------------------
## 7.4.1 均方差函数梯度
"""
L = 1/2*sum([(y_k - o_k)^2 for k in range(n)])

d(L) / d(oi) = sum([y_k - o_k for k in range(n)]) -  d(o_k)/d(o_i)

d(o_k)/d(o_i)仅当k=i时才为1， 其他点都为0， 也就是说，

==>>>>

d(L) / d(oi) = (y_i - o_i)
"""

## 7.4.2 交叉熵函数梯度
"""
softmax = e^zi / sum([e^zj for zj in p_lst])
softmax' = f'g + fg' / (g**2)
= [e^zi * sum([e^zj for zj in p_lst]) - e^zi * e^zj
/ (sum([e^zj for zj in p_lst])) ** 2

当 i = j:
可以化简为：
e^zi  / sum([e^zj for zj in p_lst])  
p - p**2 = pi(1-pj)

当 i != j 时：
e^zi ' = 0
0 - e^zi * e^zj / (sum([e^zj for zj in p_lst])) ** 2
= - pi*pj

softmax' = pi(1-pj) , i=j ; -pi*pj , i != j
"""
# 交叉熵梯度
"""
L = -sum([y_k*log(p_k)  for y_k, p_k in zip(y, p)])
直接提到最终损失函数对网络输出logits变量的偏导数，展开为
# log_a^x 导数为 1/(xlna)
d(L)/d(z_i) = -y_k * d(log(p_k)) / d(z_i)
=  -y_k * d(log(p_k)) / d(p_k) * d(p_k) / d(z_i)
= -y_k * (1/p_k) * d(p_k) / d(z_i)

softmax' = d(p_k) / d(z_i)
# k = i
-y_k * (1/p_k) * pk(1-pi) = -y_i*(1-pi)
# k != i
-y_k * (1/p_k) * (-pk*pi) = y_k * pi
# 合并

d(L)/d(z_i) = -y_i*(1-pi) + sum([y_k * pi for y_k in y_lst])
= -y_i + yi*pi + sum([y_k * pi for y_k in y_lst])

= pi(yi + sum(y_lst)) - y_i

对于分类问题中y 通过one-hot编码的方式，则有
sum(y_lst) = 1
yi + sum(y_lst) = 1 # k!=i
所以： d(L)/d(z_i) = pi - yi

"""

# 7.5 全连接层梯度
# ------------------------------------------------



