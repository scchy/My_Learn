# python3.6
# Create date: 2020-06-03
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import numpy as np

# ======== 目录 ==========
# &10 卷积神经网络
#   - 10.10 CIFAR10 与 VGG13实战
# ========================

# =======================================================================================================
#                                           第十章   卷积神经网络
# =======================================================================================================

import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
import os 
import numpy as np
from datetime import datetime
def get_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)
"""
1、log信息共有四个等级，按重要性递增为：
INFO（通知）<WARNING（警告）<ERROR（错误）<FATAL（致命的）;

2、值的含义：不同值设置的是基础log信息（base_loging），运行时会输出base等级及其之上（更为严重）的信息。具体如下：

base_loging 屏蔽信息 输出信息
'0' INFO 无 INFO + WARNING + ERROR + FATAL
'1' WARNING INFO WARNING + ERROR + FATAL
'2' ERROR INFO + WARNING ERROR + FATAL
'3' FATAL INFO + WARNING + ERROR FATAL
注意：
1、“0”为默认值，输出所有信息
2、设置为3时，不是说任何信息都不输出，ERROR之上还有FATAL
"""
# 10.10 CIFAR10 与 VGG13实战
# -------------------------------------------------
"""
CIFAR10 图片识别任务并不简单，这主要是由于CIFAR10 的图片内容需要大量细节才
能呈现，而保存的图片分辨率仅有32x32，使得部分主体信息较为模糊，甚至人眼都很难
分辨。浅层的神经网络表达能力有限，很难训练优化到较好的性能，本节将基于表达能力
更强的VGG13 网络，根据我们的数据集特点修改部分网络结构，完成CIFAR10 图片识
别。修改如下：
❑ 将网络输入调整为32x32。原网络输入为224x224，导致全连接层输入特征维度过大，
网络参数量过大
❑ 3 个全连接层的维度调整为[256, 64, 10] ，满足10 分类任务的设定
"""
## 加载CIFAR10数据集
(x, y), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()

## 删除Y的一个维度
y = tf.squeeze(y, axis=1)
y_te = tf.squeeze(y_te, axis=1)

## 构建db数据
def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255 - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def get_db(x, y, batch_=128, test_flg=False):
    db_ = tf.data.Dataset.from_tensor_slices((x, y))
    shuffle_num = x.shape[0]
    db_ = db_.shuffle(shuffle_num)
    if test_flg:
        db_ = db_.batch(batch_).map(preprocess)
    else:
        db_ = db_.batch(batch_).map(preprocess)
        db_ = db_.skip(30) # 减少内存 抽样
    return db_


tr_db = get_db(x, y)
te_db = get_db(x_te, y_te, 64, True)

 
sample = next(iter(tr_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


layers_ = [# 5 units of conv + max pooling
    # unit 1
    layers.Conv2D(64, kernel_size = [3, 3], padding = 'same', activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size = [3, 3], padding = 'same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides = 2, padding='same'),

    # unit 2
    layers.Conv2D(128, kernel_size = [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size = [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides=2, padding='same'),

    # unit 3
    layers.Conv2D(256, kernel_size = [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size = [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides=2, padding='same'),

    # unit 4
    layers.Conv2D(512, kernel_size = [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size = [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides=2, padding='same'),

    # unit 5
    layers.Conv2D(512, kernel_size = [3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size = [3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2, 2], strides=2, padding='same'),

    # flatten()
    layers.Flatten(),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10 , activation=tf.nn.softmax)

]

def main(epoch_=50):
    net_work = Sequential(layers_)
    net_work.build(input_shape=[None, 32, 32, 3])
    net_work.summary()
    optim_ = optimizers.Adam(lr=1e-4)
    tr_vars = net_work.trainable_variables

    for epoch in range(1, epoch_+1):
        print('--'*30)
        print(f'{get_now()}: start epoch: {epoch}')
        for step, (x, y) in enumerate(tr_db):
            with tf.GradientTape() as tape:
                # [b , 32, 32, 3]=> [b ,1 ,1, 512]
                logists = net_work(x)
                y_onhot = tf.cast(tf.one_hot(y, depth=10), dtype=tf.float32)
                # loss
                loss = tf.losses.categorical_crossentropy(y_onhot, logists, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, tr_vars)
            silecnc_=optim_.apply_gradients(zip(grads, tr_vars))
            
            if step % 100 == 0:
                pre_tr = tf.nn.softmax(logists)
                pre_tr = tf.cast(tf.argmax(logists, axis=1), dtype=tf.int32)
                acc_ = tf.reduce_sum(
                        tf.cast(tf.equal(pre_tr, y), dtype=tf.int32)
                    ).numpy()/x.shape[0]
                print(f'[{epoch} <{step}>], loss: {float(loss):.5f}, tr_acc: {acc_*100:.2f}%')
    
        print('开始测试')
        total_nums, total_correct = 0, 0
        for xte, yte in te_db:
            pred_y = net_work(xte)
            pred_y = tf.argmax(pred_y, axis=1)
            pred_y = tf.cast(pred_y, dtype=tf.int32)
            total_correct += tf.reduce_sum(
                tf.cast(tf.equal(pred_y, yte), dtype=tf.int32)
            ).numpy()
            total_nums += xte.shape[0]
        
        acc = total_correct / total_nums
        print(f'[ {epoch} ], te_acc: {acc*100:.2f}%')


if __name__ == '__main__':
    main()

