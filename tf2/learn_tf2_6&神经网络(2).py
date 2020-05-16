# python3.6
# Create date: 2020-05-16
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np


# 6.8 油耗预测实战
# ------------------------------------------------
## 6.8.1 数据集 
"""
数据理解
AutoMPG 数据
包含各种汽车效能指标与气缸数、重量、马力等其他因子的真实数据集

除了产地的数字代表类别其他都是数值。
1美国，2欧洲，3日本

"""
### 加载数据
def get_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    col_name = ['MPG','Cylinders','Displacement','Horsepower','Weight',
            'Acceleration', 'Model Year', 'Origin']
    df_read = pd.read_csv(url, names=col_name
                        , na_values='?', comment='\t'
                        ,sep = ' ', skipinitialspace=True)
    return df_read

dataset = get_data()

### 1- 数据查看 
dataset.isna().sum()
# Horsepower
import matplotlib.pyplot as plt
dataset['Horsepower'].plot.hist()
plt.show()
# 略有偏差 直接删除
dataset = dataset.dropna()
dataset.isna().sum()

#  Origin one-hot一下
dataset['Origin'] = dataset['Origin'].map({
    1:'USA', 2:'Euro', 3:'Japan'
})
tmp_df = pd.get_dummies(dataset['Origin'])
dataset = pd.concat([dataset.iloc[:,:-1], tmp_df], axis=1)

# 2- 查看特征是否相关
dataset.columns
target = 'MPG'
num_col = [ 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
       'Acceleration', 'Model Year']

import seaborn as sns
n=0
for col in num_col:
    sns.lmplot(x=col, y = target, data=dataset )
    n += 1
plt.show()

# 3- 拆分训练集测试集
train_set = dataset.sample(frac = 0.8, random_state = 0)
test_set = dataset.drop(train_set.index)
## 将目标变量输出
train_labels = train_set.pop('MPG')
test_labels = test_set.pop('MPG')
## 查看训练集的输入x的统计数据
train_stats = train_set.describe().T
def norm(x):
    # 会一一对应的做差
    return ( x - train_stats['mean']) / train_stats['std']

norm_tr = norm(train_set)
norm_te = norm(test_set)

print(norm_tr.shape, norm_te.shape)


# 4- 构建tf训练数据集
train_db = tf.data.Dataset.from_tensor_slices((norm_tr.values, train_labels.values))
train_db = train_db.shuffle(314).batch(32) # 随机打散，批量化

test_db = tf.data.Dataset.from_tensor_slices((norm_te.values, test_labels.values))
test_db = test_db.shuffle(78).batch(32) # 随机打散，批量化

## 6.8.2 创建网络
"""
考虑到AutoMp数据集规模较小，我们只创建一个3层的全连接层网络来完成
MPG值的预测任务。输入X的特征共9种。
第一层、第二层的输出节点设计为64， 64。由于只有一种预测值，输出层
输出节点设计为1，考虑MPG <- R ，因此可以不加激活函数，也可以用Relu
"""
class Network(tf.keras.Model):
    # 回归网络
    def __init__(self):
        super(Network, self).__init__()
        # 创建三个全连接层
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        # 依次通过三个全连接层
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

## 6.8.3 训练与测试
model = Network()
## 4为任意的batch数量，9为特征长度
model.build(input_shape = (4, 9))
model.summary() # 打印网络信息
## 创建优化器，指定学习率
optimizer = tf.keras.optimizers.RMSprop(0.001)

## 训练
from datetime import datetime
def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

epoch_lst = []
tr_loss_lst, te_loss_lst = [], []
for epoch in range(1, 201):
    for step, (x, y) in enumerate(train_db):
        # 梯度记录器
        # print(x.shape, x)
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(tf.keras.losses.MSE(y, out))
            mase_loss = tf.reduce_mean(tf.keras.losses.MAE(y, out))
            

    grads = tape.gradient(loss, model.trainable_variables)
    # 更新梯度
    drop_msg = optimizer.apply_gradients(zip(grads, model.trainable_variables))#, read_values=False)

    if (epoch >= 10) and (epoch % 10 == 0):
        loss_all = 0
        for x, y in test_db:
            x_tp = model.call(x)
            losst = tf.reduce_mean(tf.keras.losses.MSE(y, out))
            loss_all += losst

        tr_loss_lst.append(loss.numpy())
        te_loss_lst.append(loss_all)
        epoch_lst.append(epoch)
        print('\n','='*60)
        print(f'{get_now()}: 迭代到epoch: {epoch} ， tr_mse_loss: {loss.numpy()} ,te_mse_loss: {loss_tf}')
 


x_ = epoch_lst #list(range(len(tr_loss_lst)))
fig, axes = plt.subplots(figsize=(8, 4))
axes.plot(x_, tr_loss_lst, label='tr')
axes.plot(x_, te_loss_lst, label='te')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

"""
可以看出 100 个epoch后几乎就没有提升了
"""
