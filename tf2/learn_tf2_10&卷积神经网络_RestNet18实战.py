# python3.6
# Create date: 2020-06-13
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import numpy as np

# =======================================================================================================
#                                           第十章   卷积神经网络——CIFAR10与RestNet18实战
# =======================================================================================================
# 10.14 CIFAR10 与 ResNet18实战
#---------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, optimizers, losses

class ResBlock(layers.Layer):
    """
    残差模块
    2卷积 一跳跃
    """
    def __init__(self, filternum, strides=1):
        super(ResBlock, self).__init__()
        # Conv1
        self.conv1 = layers.Conv2D(filternum, kernel_size=3, strides=strides, padding='same', name='resblock_conv1')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # Conv2 strides=1 是确定的 
        self.conv2 = layers.Conv2D(filternum, kernel_size=3, strides=1, padding='same', name='resblock_conv2')
        self.bn2 = layers.BatchNormalization()

        if strides != 1:
            self.jump = Sequential()
            self.jump.add(layers.Conv2D(filternum, kernel_size=1, strides=strides))
        else:
            self.jump = lambda x:x
        
    def call(self, inputs, training=True):
        #[b h w c]
        # Conv1
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # Conv2 
        out = self.conv2(out)
        out = self.bn2(out)
        # jump
        identity = self.jump(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output



class RestNet(Model):
    """
    RestNet
    """
    def __init__(self, layers_dims, num_classes=10):
        super(RestNet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64, kernel_size=3, strides=1), 
            layers.BatchNormalization(), 
            layers.Activation('relu'), 
            layers.MaxPooling2D(pool_size=2, strides=1, padding='same')
        ])
        # 堆叠4个 Restblock # sum(2*layers_dims)
        self.layer1 = self.build_resblock(64, layers_dims[0])
        self.layer2 = self.build_resblock(128, layers_dims[1], strides=2)
        # 电脑内存问题 改为2个
        # self.layer3 = self.build_resblock(256, layers_dims[2], strides=2)
        # self.layer4 = self.build_resblock(512, layers_dims[3], strides=2)
        # 通过Pooling层将高宽降低为1*1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后一个全连接层
        self.fc = layers.Dense(num_classes)
    
    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    
    def build_resblock(self, filter_num, blocks, strides=1):
        res_block = Sequential()
        # 只有第一个restblocak步长可能不是1， 进行下采样
        res_block.add(ResBlock(filter_num, strides=strides))
        for _ in range(1, blocks):
            res_block.add(ResBlock(filter_num, strides=1))
        
        return res_block


def restnet18(num_classes):
    return RestNet([2, 2], num_classes)



# 一、 数据加载 
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
y_tr = tf.squeeze(y_tr, axis=1)
y_te = tf.squeeze(y_te, axis=1)

def cifarpreprocess(x, y):
    """
    预处理
    """
    x = 2 * tf.cast(x, dtype=tf.float32)/255.0 - 1
    y_onhot = tf.one_hot(tf.cast(y, dtype=tf.int32), depth=10)
    return x, tf.cast(y_onhot, dtype=tf.float32)


def cifarget_db(x, y, batch_=100, tr_flg=False):
    db_ = tf.data.Dataset.from_tensor_slices((x, y))
    shuffle_num = x.shape[0]
    db_ = db_.shuffle(shuffle_num)
    if tr_flg:
        db_ = db_.skip(200)

    db_ = db_.map(cifarpreprocess).batch(batch_)
    return db_

def corecct(y, y_pred):
    y_ = tf.cast(tf.argmax(y, axis=1), dtype=tf.int32)
    y_pred_ = tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.int32)
    coreccts = tf.reduce_sum(tf.cast(tf.equal(y_pred_, y_), dtype=tf.int32))
    return coreccts



tr_db = cifarget_db(x_tr, y_tr, batch_=200, tr_flg=True)
te_db = cifarget_db(x_te, y_te, batch_=200)

model = restnet18(num_classes=10)
model.build(input_shape=(None,32,32,3))
model.summary()
optim_ = optimizers.Adam(lr=1e-4)

dir(optimizers)

# acc_lst_tr = []
# loss_tr = []
# acc_lst_te = []
# loss_te = []

for epoch in range(50):
    for step, (x, y) in enumerate(tr_db):
        with tf.GradientTape() as tape:
            out = model(x)
            # 计算交叉熵
            loss = tf.losses.categorical_crossentropy(y, out, from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        silence_ = optim_.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            c = corecct(y, out)
            acc_ = c / x.shape[0]
            print(f'[ {epoch}  {step} ] loss_tr:{loss:.5f}   acc_tr: {acc_.numpy()*100:.2f}%')


    print('开始测试....')
    c_total, loss_total = 0, 0
    n = 0
    for xt, yt in tr_db:
        yt_p = model(xt)
        losst = tf.losses.categorical_crossentropy(y, out, from_logits=True)
        loss_total += tf.reduce_mean(losst)
        c_total += corecct(y, out)
        n += xt.shape[0]
    accte = c_total/n
    print(f'[ {epoch} ] acc_te: {accte.numpy()*100:.2f}%')

