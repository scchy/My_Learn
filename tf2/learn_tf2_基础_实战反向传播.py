# python 3.6
# tf2.0
# Author:Scc_hy
# Create date: 2020-05-08
# Refrence: https://www.icode9.com/content-4-601528.html

## 4.10 前向传播实战
### 三层神经网络的实现
### out = relu{relu{relu[X@W1 + b1]@w2 + b2}@W3 + b3}
#-----------------------------------------------
"""
采用MNIST手写数字图片集，输入节点数为784，
第一层的输出节点数是256， 第二层的输出节点是128，第三层的输出节点是10，也就是当前样本属于10类别的概率

"""
# 0- 数据读取
import tensorflow as tf
from tensorflow.keras import datasets

(x, y), (x_val, y_val ) = datasets.mnist.load_data()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.     #先将类型转化为float32，再归一到0-1
    x = tf.reshape(x, [-1, 28*28])              #不知道x数量，用-1代替，转化为一维784个数据
    y = tf.cast(y, dtype=tf.int32)              #转化为整型32
    y = tf.one_hot(y, depth=10)                 #训练数据所需的one-hot编码
    return x, y


train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(60000)      #尽量与样本空间一样大
train_db = train_db.batch(100)          #128
train_db = train_db.map(preprocess)


test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.shuffle(10000)      #尽量与样本空间一样大
test_db = test_db.batch(100)          #128
test_db = test_db.map(preprocess)



# 0- 定学习率
lr = 0.003
loss_ls = []
loss_te = []
precision_lst = []

# 1- 首先创建每个非线性函数的w, b 参数张量 初始化
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1), name='w1')
b1 = tf.Variable(tf.zeros([256]), name='b1')
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1), name='w2')
b2 = tf.Variable(tf.zeros([128]), name='b2')
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1), name='w3')
b3 = tf.Variable(tf.zeros([10]), name='b3')

# 
for i in range(25):
    for step, (x, y_onehot) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # 3- 完成非线性函数
            ## 3-1 
            h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)

            ## 3-2 [b, 256] => [b, 128]
            h2 = h1@w2 + tf.broadcast_to(b2, [x.shape[0], 128])
            h2 = tf.nn.relu(h2)

            ## 3-3  [b, 128] => [b, 10]
            out = h2@w3 + b3 
            # 将真实的标注张量y转变为one-hot编码， 并计算与out均方差
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)

        # 通过 tape.gradient() 函数求得网络参数到梯度信息
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 通过偏导更新数据 theata' = theata - lr * eL / etheata
        w1.assign_sub(lr * grads[0], read_value = False)
        b1.assign_sub(lr * grads[1], read_value = False)
        w2.assign_sub(lr * grads[2], read_value = False)
        b2.assign_sub(lr * grads[3], read_value = False)
        w3.assign_sub(lr * grads[4], read_value = False)
        b3.assign_sub(lr * grads[5], read_value = False)
        
        if (step >= 500) and (step % 500 == 0):
            total, total_correct, lossts = 0., 0., []
            for xt, yt in test_db:
                h1t = xt@w1 + tf.broadcast_to(b1, [xt.shape[0], 256])
                h1t = tf.nn.relu(h1t)
                h2t = h1t@w2 + tf.broadcast_to(b2, [xt.shape[0], 128])
                h2t = tf.nn.relu(h2t)
                outt = h2t@w3 + b3 
                # 记录测试集损失
                losst = tf.square(yt - outt)
                losst = tf.reduce_mean(losst)
                lossts.append(losst.numpy())

                pred = tf.argmax(outt, axis = 1) 
                yt1 = tf.argmax(yt, axis = 1) 
                total += outt.shape[0]
                
                t_p = tf.equal(pred, yt1)
                total_correct += tf.reduce_sum(tf.cast(t_p, dtype=tf.int32)).numpy()

            acc = total_correct / total
            precision_lst.append(acc)
            loss_ls.append(loss.numpy())
            loss_te.append(np.mean(lossts))
            print('\n','--'*20)
            print(f'{get_now()}: 迭代到第 {i}次-step:{step} ， tr_loss: {round(loss_ls[-1], 4)} ,te_loss: {round(loss_te[-1], 4)}, 准确率： {acc}')


ln_ = list(range(len(loss_ls)))
import matplotlib.pyplot as plt 
fig, axes = plt.subplots(figsize=(8, 6))

axes.plot(ln_, loss_ls, label='train')
axes.plot(ln_, loss_te, label='test', linestyle='--')
axes.legend()
axes_ = axes.twinx()
axes_.plot(ln_, precision_lst, label='acc')
axes_.legend()

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()




