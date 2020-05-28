# python3.6
# Create date: 2020-05-27
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &10 卷积神经网络
#   - 10.1 全连接网络的问题
#    -- 10.1.3 卷积
#   - 10.2 卷积神经网络
#   - 10.3 卷积层实现
#    -- 10.3.2 卷积层类
# ========================

# =======================================================================================================
#                                           第十章   卷积神经网络
# =======================================================================================================


# 10.1 全连接网络的问题
# ---------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets 
# 4层网络
model = Sequential(
    [
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='relu'),
    ]
)
model.build(input_shape=(4, 784))
model.summary()
# 参数计算 例：
"""
in -> h1 ->h2 -> h3 -> h4(out)
h1
784 * 256 + 256 = 200960
h2 
256 * 256 + 256 = 65792
"""
"""
如果不设置显存占用方式，那么默认会占用全部显存。
使用tf的显存使用方式设置为按需分配：
"""
gpus = tf.config.experimental.list_physical_devices('GPU')

# 需要 beta1版本
dir(tf.config.experimental)
if gpus:
    try: 
        # 设置GPU显存占用为按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'{len(gpus)} Pysical GPUs, {len(logical_gpus)} Logical GPUs.')
    except RuntimeError as e:
        print(e)
"""
在batch siez设置32的情况下，我们观察到显存占用约708MB,
内存占用约870MB，由于现代深度学习框架设计考量不一样，
这个数字仅做参考。
"""

## 10.1.1 局部相关性
"""
输出节点是否有不要和全部的输入节点相连接呢？
可以考虑输入节点对输出节点的重要性分布，选出最重要的
一部分输入节点，而抛弃重要性较低的部分节点，这样输出节点只需要与输出最重要 
的一部分输入节点相连接
"""
"""
oj = sigma(sum([oi * wij + bj])
其中 仅仅选择 重要性最高的K个节点集合。
可以将全连接层的权值从 I * J -->> k * J 

那么核心步骤就是探索I层输入节点对于j层输出节点的重要性分布。

如果见到认为与当前像素的欧式距离小于 k/sqrt(2)
的像素点重要性较高， 欧式距离大于 k/sqrt(2) 重要性较低

这个高宽为k的窗口 称为感受野，它表征了每个像素对于中心
像素的重要性分布情况，网格内的像素才会被考虑，
网格外的像素对于中心像素会被简单的忽略

基于距离的重要性分布假设称为局部相关性，它只关注和自己距离较近的部分节点而忽略距离较远的节点，
在这种重要性分布假设下，全连接层的连接模式变为
输出节点j只与以j为中心的局部区域(感受野)相联系

此时网络层的输入输出关系表达如下：
oj =sigma(sum([ wij*xi+bj for i,j in set_dist(i,j)<= k/sqrt(2) ]))
"""

## 10.1.2 权值共享
"""
当前层的参数量为 k*k * J 
一般 k*k << I

再者, 对于每个输出节点oj，均使用相同的权值矩阵W
，那么无论输出节点的数量J，网络层的参数总是k*k。
"""
"""
w = [
[w00, w01, w02],
[w10, w11, w12],
[w20, w21, w22]
]
与对应感受野内的像素相乘累计加， 做为左上角像素的输出值；
在计算右下方感受野区域时，共享权值参数W，即使用相同的权值参数W相乘累加，得到右下角的像素输出
此时完成层的参数只有 3*3 = 9， 且与输入、输出节点数无关。

"""
## 10.1.3 卷积
"""
这种权值相乘累加的运算其实是信号处理邻域的一种标准运算：
离散卷积运算。离散卷积运算在计算机视觉中有广泛的应用

1D连续信号的卷积运算被定义2个函数的积分：函数f(x)
g(x). 其中g(x)经过翻转g(-x)和平移后变成g(n-x).卷积的'卷'
是指翻转平移操作，'积'是指积分运算，ID连续卷积定义为：


(f * g)(n) = f(x)g(n-x)dx
-->> 离散卷积将积分变成累加运算

(f * g)(n) = sum([f(x)g(n-x)])
"""
"""
2D 图片函数 f(m, n) , 卷积核 g(m, n)
其中f, g仅仅在个字窗口有效区域存在值，其他区域视为0

[f * g](m, n) = f(i, j)g(m-i, n-j)
f(m, n) = [ # m *n
    [2, 3, 1],
    [0, 5, 1],
    [1, 0, 8]
]

g(m, n) = [ # m *n
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]
"""
"""离散卷积运算
f与 g翻转移动后的元素乘积
f 卷积运算后

f_aft = [
[[f*g](-1, -1), [f*g](0, -1), [f*g](1, -1) ],
[[f*g](-1, 0), [f*g](0, 0), [f*g](1, 0) ],
[[f*g](-1, 1), [f*g](0, 1), [f*g](1, 1) ]
]
"""
import numpy as np
def smpl_fg(f, g_turn, m_n):
    """
    仅限 3*3
    param f：原图片 np.array图片
    param g_turn: 翻转后的核
    """
    m_max, n_max = f.shape
    m_n_i = m_n - 1 
    m, n = m_n_i
    m = m_max if m == 0 else m 
    n = n_max if n == 0 else n 
    if m_n[0] + 1 == m_max and m_n[1] + 1 == n_max:
        sum_a = g_turn[:-1*m, :-1*n] * f[m:, n:]
    elif m_n[0] + 1 == m_max:
        sum_a = g_turn[:-1*m, -1*n:] * f[m:, :n]
    elif m_n[1] + 1 == m_max:
        sum_a = g_turn[-1*m:, :-1*n] * f[:m, n:]
    else:
        sum_a = g_turn[-1*m:, -1*n:] * f[:m, :n]
    return sum(sum_a.ravel())

def smpl_fg_array(f, g_turn):
    m, n = f.shape
    out = np.zeros_like(f)
    m_idx = np.repeat(np.arange(m), n).reshape((m,n))
    n_idx = np.array(list(range(n))*m).reshape((m,n))
    for  m_line, n_line in zip(m_idx, n_idx):
        for m, n in zip(m_line, n_line):
            out[m, n] = smpl_fg(f, g_turn, np.array([m, n]))
    return out

f= np.array([ # m *n
    [2, 3, 1],
    [0, 5, 1],
    [1, 0, 8]
])

g_turn = np.array([ # m *n
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])


smpl_fg(f, g_turn, np.array([0, 2]))
smpl_fg_array(f, g_turn)



# 10.2 卷积神经网络
# ---------------------------------------------------
## 10.2.1 单通道输入，单卷积核
# p223
## 10.2.2 多通道输入，单卷积核
"""
三通道 分别做单通道的操作 再加权

一个固定的卷积核，只能完成某种逻辑的特征提取
"""

## 10.2.3 多通道输入，多卷积核
## 10.2.4 步长
"""
步长可以控制信息密度。
小步长获取更多的特征；大步长，利用减少计算代价，过滤冗余信息，输出张量的尺寸也更小
"""
## 10.2.5 填充
"""
有时候希望输出的高宽和输入的X的高宽相同。 需要对X进行填充。
填充后再运算

神经网络层的输出尺寸
[b, h', w', C_out] 由于卷积核的数量C_out, 卷积核的大小k，步长s，填充数p确定

当ph = pw 时， x输入 h w 
h' = (h + 2ph - k)/s + 1 
w' = (w + 2pw - k)/s + 1 

"""

# 10.3 卷积层实现
# ---------------------------------------------------
"""
tf.nn.conv2d 可以方便实现2D卷积运算。
tf.nn.conv2d 基于输入x [b,h,w,c] 和 卷积核 w:[k, k, c_in, c_out]
得到输出 O [b, h', w', c_out]
"""
x = tf.random.normal([2, 5 ,5, 3]) # 模拟输入 
w = tf.random.normal([3, 3, 3, 4]) # k,k,c_in, c_out 对应3通道 4个 3*3核
# 步长1 padding 为0
# padding=[[0,0],[上,下],[左,右],[0,0]]
out = tf.nn.conv2d(x, w, strides=1, padding=[[0,0], [0,0], [0,0],[0,0]])
out.shape # 5 + 0 - 3 + 1
out = tf.nn.conv2d(x, w, strides=1, padding=[[0,0], [1,1], [1,1],[0,0]])
out.shape # 5 +1*2 -3 + 1
# 设置输入与输出同大小，仅stride=1时可用
out = tf.nn.conv2d(x, w, strides=1, padding='SAME')
out.shape
"""
当s>1，设置padding='SAME' 将使得输出高、宽将变成1/s蓓的减少
"""
# h w先padding到可以被整除的 6 -> 6/3
out = tf.nn.conv2d(x, w, strides=3, padding='SAME')
out.shape

# 设置偏置 
b = tf.zeros([4])
out = out + b

## 10.3.2 卷积层类
