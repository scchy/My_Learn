# python3.6
# Create date: 2020-05-30
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import numpy as np

# ======== 目录 ==========
# &10 卷积神经网络
#   - 10.4 LeNet-5实战
# ========================

# =======================================================================================================
#                                           第十章   卷积神经网络
# =======================================================================================================
# 10.4 LeNet-5实战
# -------------------------------------------------

### 完整流程
import tensorflow as tf
from tensorflow.keras import layers, Sequential, losses, optimizers
from tensorflow.keras import datasets

## 一、数据加载
(x, y), (x_val, y_val ) = datasets.mnist.load_data()
print(f'x.shape: {x.shape}, x_val.shape: {x_val.shape}')

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255
    # 作为图像数据 增加一个通道
    x = tf.expand_dims(x, axis=3)
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

def get_db(x, y):
    db_ = tf.data.Dataset.from_tensor_slices((x, y))
    suffle_num = x.shape[0]
    db_ = db_.shuffle(suffle_num)
    db_ = db_.batch(100)
    db_ = db_.map(preprocess)
    return db_

train_db = get_db(x, y)
test_db = get_db(x_val, y_val)


## 二、搭建网络
network = Sequential([
    # 卷积层一 6 3*3
    layers.Conv2D(6, kernel_size=3, strides=1, name='conv1_6_3X3'),
    layers.MaxPooling2D(pool_size=2, strides=2, name='pool1_half'),
    layers.ReLU(),
    # 卷积层2 16 3*3
    layers.Conv2D(16, kernel_size=3, strides=1, name='conv2_16_3X3'),
    layers.MaxPooling2D(pool_size=2, strides=2, name='pool2_half'),
    layers.ReLU(),
    layers.Flatten(),
    # 全连接侧
    layers.Dense(120, activation='relu', name='fc1_120'),
    layers.Dense(84, activation='relu', name='fc2_84'),
    layers.Dense(10, name='fc3_10_noactive'),
])

network.build(input_shape=(4, 28, 28, 1))
network.summary()

# 三、增加评估器、损失函数
optim_ = optimizers.RMSprop(0.001)
loss_func = losses.CategoricalCrossentropy(from_logits=True)

# gpus = tf.config.experimental.list_physical_devices('GPU')

# with tf.device('/gpu:0'):

# 四、模型训练
# with tf.device('/gpu:0'):
tr_loss_lst, te_loss_lst, acc_lst = [], [], [] 
for epoch in range(1, 5): # 50个epoch训练
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = network(x)
            loss = loss_func(y, out)
        # 获取需要计算梯度的参数 和损失函数
        grads = tape.gradient(loss, network.trainable_variables)
        # 自动更新梯度学习率 0.01 类似 grad.subssign(grads[0], network.trainable_variables[0], reac_values=False)
        silence_ = optim_.apply_gradients(zip(grads, network.trainable_variables))
        
        if step % 100 == 0:# 50step检查一次
            correct, total, loss_te = 0, 0, 0
            n = 0
            for x, y in test_db:
                out_i = network(x)
                out_pred = tf.argmax(out_i, axis=1)
                yi_flatten =tf.argmax(y, axis=1) 
                loss_te += loss_func(y, out_i).numpy()
                correct += tf.reduce_sum( tf.cast( tf.equal(yi_flatten, out_pred), tf.int32 )).numpy()
                total += x.shape[0]
                n += 1
            acc_ = correct/total
            tr_loss_lst.append( loss.numpy() )
            te_loss_lst.append( loss_te/n )
            acc_lst.append(acc_)
            print(f'epoch: {epoch}, step: {step} ......... ')
            print(f'te_loss: {tr_loss_lst[-1]:.5f}, te_loss: {te_loss_lst[-1]:.5f}, te_acc: {acc_lst[-1]*100:.2f}', '\n')


# 五、查看损失 和准确性
import matplotlib.pyplot as plt

x_ = list(range(len(tr_loss_lst)))
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].plot(x_, tr_loss_lst, c='steelblue', alpha=0.8, label = 'tr')
axes[0].plot(x_, te_loss_lst, c='darkred', alpha=0.8, label = 'te')
plt.legend()
axes[1].plot(x_, acc_lst, label = 'acc')
plt.legend()
plt.show()

