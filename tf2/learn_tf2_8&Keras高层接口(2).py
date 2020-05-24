# python3.6
# Create date: 2020-05-21
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &8 Keras高层接口
#   - 8.4 自定义类
#   - 8.5 模型乐园
##### keras.applications子模块下
#   - 8.6 测量工具
#   - 8.7 可视化 TensorBoard

# ========================

# =======================================================================================================
#                                           第八章   Keras高层接口
# =======================================================================================================


# 8.4 自定义类
# ---------------------------------------------------
"""
在创建自定义神经网络层时，需要几层子layres.Layer基类；创建
自定义的网络类，需要继承自keras.Model基类，这样产生的自定义类才能够
方便利用Layers/Model基类提供的参数管理功能，同时也能够与其他的标准网络
层交互使用
"""
## 8.4.1 自定义网络层
import tensorflow as tf # 
from tensorflow.keras import layers
"""
由于是全连接层，所以需要设置inp_dim, outp_dim
并通过self.add_variable(name, shape)并设置为需要优化

"""
class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        # 创建权值张量并添加类管理列表中，设为需要优化
        self.kernel = self.add_variable('w', [inp_dim, outp_dim]
                                        , trainable=True)


    def call(self, inputs, training=None):
        """
        完成 sigma(X@W)
        """
        out = inputs@self.kernel
        return tf.nn.relu(out)



net = MyDense(4, 3)
net.variables, net.trainable_variables


## 8.4.2 自定义网络层
#### 完成上述无偏置的全连接层，开实现MNIST手写数字图片模型
#### 的创建
network = tf.keras.Sequential(
    [MyDense(256, 128),
    MyDense(128, 64), 
    MyDense(64, 32), 
    MyDense(32, 10)]
)
network.build(input_shape=(None, 28*28))
network.summary()

# 更普遍的做法
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x  


# 8.5 模型乐园
# keras.applications子模块下
# ---------------------------------------------------
## 8.5.1 加载模型
"""
以ResNet50迁移学习为例，一般将ResNet50去掉最后一层后的网络作为新任务的特征提取子网络，
即利用ImageNet上预训练的特征提取方法迁移到我们自定义的数据集上，
并根据自定义的任务追加一个对应数据类别数的全连接分类层，从而可以在预训练网络的基础上
可以快速高效的学习新任务。首先利用Keras模型乐园加载ImageNet预训练的ResNet50网络
"""
# 加载ImageNet预训练网络模型
resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
resnet.summary()
x = tf.random.normal([4, 224, 224, 3])
out = resnet(x)
"""
从服务器下载网络结构和ImageNet数据集上训练好的网络参数，
去掉最后一层，网络的输出大小为 [b, 7,7,2048].
对于某个具体的任务，需要设置自定义的输出节点，
以100类的分类任务为例，我们在RestNet50基础上重新构建网络。
新建一个池化层(这里的池化层可以理解为维度缩减功能)，将
特征从[b, 7, 7, 2048]降维到[b, 2048]
"""
globale_average_layer = layers.GlobalAveragePooling2D()
# 利用上层的输出作为本层的输入，测试其输出
x = tf.random.normal([4, 7, 7, 2048])
out = globale_average_layer(x) # 池化降维
# 新建一个全连接层，并设置输出节点数为1000
fc = layers.Dense(100)
x = tf.random.normal([4, 2048])
out = fc(x)
mynet = tf.keras.Sequential(
    [resnet, globale_average_layer ,fc]
)


# 8.6 测量工具
# kears.metrics
# ---------------------------------------------------
## 8.6.1 新建测量器
"""
我们以统计误差值为例， 在向前运算时，我们会得到一个batch的平均误差
但是我们希望统计epoch的平均误差，所以需要用Mean()测量器
"""
loss_meter = tf.keras.metrics.Mean()

## 8.6.2 写入数据
loss_meter.update_state(float(loss))

## 8.6.3 读取统计信息
print(setp, 'loss:', loss_meter.result())

## 8.6.4 清除
"""
每次读取完平均误差后，清零统计信息后，以便下一轮
统计开始
"""
loss_meter.reset_states()


## 8.6.5 准确率统计实战
acc_meter = metrics.Accuracy()
"""
在每次前向计算完成后，记录训练准确率。需要注意的是，Accuracy类的update_state函数
的参数为预测值和真实值，而不是已经计算过的batch的准确率
"""
# [b, 784] -> [b, 10]
out  = network(x)
# [b, 10] => [b] 经过argmax计算
pred = tf.argmax(out, axis=1)
pred = tf.cast(pred, dtype=tf.int32)
acc_meter.update_state(y, pred)

# 读取统计结果
print(step, 'Evaluate Acc:', acc_meter.result().numpy())
acc_meter.reset_states()


# 8.7 可视化 TensorBoard
# ---------------------------------------------------
"""
通过TensorFlow将监控数据写入到文件系统，并利用
Web后端监控对应的文件目录，从而可以允许用户从远程
查看网络的监控数据
"""
## 8.7.1 模型端
"""
在模型端，需要创建写入监控数据的Summary类，并在需要的时候
写入监控数据。首先通过tf.summary.create_file_write创建监控对象。
并指定监控数据的写入目录

"""
summary_writer = tf.summary.create_file_write(log_dir)
# 通过summary_writer函数记录监控数据，并指定时间戳step
with summary_writer.as_default():
    # 当前时间戳step上的数据为loss, 写入到ID位train-loss对象中
    tf.summary.scalar('train-loss', float(loss), step=step)

with summary_writer.as_default():
    # 写入测试准确率
    tf.summary.scalar('test-acc', float(total_correct/total), steo=step)
    # 可视化测试用的图片，设置最多可视化9张图片
    tf.summary.image('val-onebyone-images:', val_images, max_outputs=9, step=step)

## 8.7.1 浏览器端

