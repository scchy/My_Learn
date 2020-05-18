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



