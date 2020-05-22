# python3.6
# Create date: 2020-05-11
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &5 tf进阶
#   - 5.7 经典数据加载
#   - 5.8 MNIST 测试实战

# ========================

# =======================================================================================================
#                                           第五章   TensorFlow进阶
# =======================================================================================================


# 5.7 经典数据加载
# ------------------------------------------------

"""
keras.datasets 模块提供了常用经典数据集的自动下载、管理、加载与转换功能，
并且提供了 tf.data.Dataset数据集对象， 方便实现多线程(Multi-thread)， 预处理（Preprocess）， 
,Shuffle, 批训练

"""
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import datasets 
# 加载MNIST数据
(x, y), (x_te, y_te) = datasets.mnist.load_data()

x.shape, y.shape, x_te.shape, y_te.shape
train_db = tf.data.Dataset.from_tensor_slices((x, y))

## 5.7.1  随机打散数据
train_db = train_db.shuffle(x.shape[0])
## 5.7.2  批训练
train_db = train_db.batch(128) # 并行计算128个样本数据
## 5.7.3  预处理
def preprocess(x, y):
    """
    预处理x , y
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28]) # 展平数据 
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return  x, y

train_db = train_db.map(preprocess)

## 5.7.4 循环训练
for epoch in  range(20):
    for step, (x, y) in enumerate(train_db):
        print('....')

# 或者
# 使得 for x, y in train_db 循环迭代20个epoch才会退出。 
train_db = train_db.repeat(20)



# 5.8 MNIST 测试实战
# ------------------------------------------------
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import datasets 
import numpy as np
from datetime import datetime
def get_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 一、加载数据
(x, y), (x_te, y_te) = datasets.mnist.load_data()

train_db = tf.data.Dataset.from_tensor_slices((x, y))
test_db = tf.data.Dataset.from_tensor_slices((x_te, y_te))

## 1.1 预处理
def preprocess(x, y):
    """
    预处理x , y
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28]) # 展平数据 
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return  x, y

train_db = train_db.shuffle(60000)      #尽量与样本空间一样大
train_db = train_db.batch(100)        
train_db = train_db.map(preprocess)
 
test_db = test_db.shuffle(10000)      #尽量与样本空间一样大
test_db = test_db.batch(100)          
test_db = test_db.map(preprocess)


# 二、准备初始网络参数
lr = 0.003
loss_tr = []
loss_te = []
acc_lst = []

# 增加stddev 使得在其在0 附近，太大容易直接nan 太小迭代过慢
w1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1), name = 'w1')
b1 = tf.Variable(tf.zeros([256]), name = 'b1')
w2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1), name = 'w2')
b2 = tf.Variable(tf.zeros([128]), name = 'b2')
w3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1), name = 'w3')
b3 = tf.Variable(tf.zeros([10]), name = 'b3')


# 三、迭代学习
for epoch in range(26): ## 迭代26次
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)   
            out = h2@w3 + b3 

            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)
        
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 通过偏导更新数据 theata' = theata - lr * eL / etheata
        w1.assign_sub(lr * grads[0], read_value = False)
        b1.assign_sub(lr * grads[1], read_value = False)
        w2.assign_sub(lr * grads[2], read_value = False)
        b2.assign_sub(lr * grads[3], read_value = False)
        w3.assign_sub(lr * grads[4], read_value = False)
        b3.assign_sub(lr * grads[5], read_value = False)

        if (step >= 500) and (step%500 == 0):
            lossts = []
            corr_sum, loss_sum, totals = 0, 0, 0
            for x_te, y_te in test_db:
                h1t = x_te@w1 + b1
                h1t = tf.nn.relu(h1t)
                h2t = h1t@w2 + b2
                h2t = tf.nn.relu(h2t)   
                outt = h2t@w3 + b3 

                losst = tf.square(y_te - outt)
                losst = tf.reduce_mean(losst)
                lossts.append(losst.numpy())

                predict_ = np.argmax(outt, axis=1) 
                y_te_ = np.argmax(y_te, axis=1) 
                corr_sum += tf.reduce_sum(tf.cast(tf.equal(predict_, y_te_), dtype = tf.int32))
                totals += outt.shape[0]
            
            loss_tr.append(loss.numpy())
            loss_te.append(np.mean(lossts))

            acc_ = corr_sum / totals
            acc_lst.append(acc_)
            print('\n','--'*20)
            print(f'{get_now()}: 迭代到第 {epoch + 1}次-step:{step} ， tr_loss: {round(loss_tr[-1], 4)} ,te_loss: {round(loss_te[-1], 4)}, 准确率： {acc_}')

# 四、评估曲线
import matplotlib.pyplot as plt 
fig, axes = plt.subplots(figsize=(10, 5))

x_plot = list(range(1, len(loss_tr)+1))
axes.plot(x_plot, loss_tr, linestyle = '--', marker = '*' ,label = 'tr_loss')
# axes.plot(x_plot, loss_te, linestyle = '--', label = 'te_loss')
axes.legend()

axe = axes.twinx()
axe.plot(x_plot, acc_lst, c='orange', marker='o', label = 'acc_', alpha = 0.6)
axe.legend()
plt.show()

