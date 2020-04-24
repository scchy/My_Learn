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


