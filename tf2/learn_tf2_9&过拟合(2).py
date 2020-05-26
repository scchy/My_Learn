
# python3.6
# Create date: 2020-05-26
# Function: 19年11月最新-《TensorFlow+2.0深度学习算法实战教材》

import tensorflow as tf 
import math
import abc
import numpy as np

# ======== 目录 ==========
# &9 过拟合
#   - 9.8 过拟合问题实战
# ========================

# =======================================================================================================
#                                           第九章   过拟合
# =======================================================================================================

# 9.8 过拟合问题实战
# ---------------------------------------------------
## 9.8.1 构建数据集
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

n_sample = 1000
test_rate = 0.3 
x, y = make_moons(n_sample, noise=0.2, random_state=100)

x_tr, x_te, y_tr, y_te = train_test_split(x, y ,test_size = test_rate, random_state=42)
def make_plot(x, y , title='', xx=None, yy=None, preds=None):
    fig, axes = plt.subplots(figsize=(10, 6))
    axes.scatter(x[y==1, 0], x[y==1, 1], label = 'y=1', c='darkred', alpha=0.7, s = 8)
    axes.scatter(x[y==0, 0], x[y==0, 1], label = 'y=0', c='steelblue', alpha=0.7, s = 8)
    plt.legend()
    plt.title(title)
    if xx is not None and yy is not None and preds is not None: 
        plt.contourf(xx, yy, preds.reshape(xx.shape), 25, alpha=0.08, cmap=plt.cm.Spectral)
        plt.contour(xx, yy, preds.reshape(xx.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    return axes
    

ax = make_plot(x, y)
plt.show()

def pre_border_dt():
    # 绘制不同层数的网络决策边界曲线
    # 可视化的 x 坐标范围为[-2, 3]
    x1_ = np.arange(-2, 3, 0.01)
    # 可视化的 y 坐标范围为[-1.5, 2]
    x2_ = np.arange(-1.5, 2, 0.01)
    # 生成 x-y 平面采样网格点，方便可视化
    xx1, xx2 = np.meshgrid(x1_, x2_)
    return xx1, xx2




from tensorflow.keras import layers, Sequential, regularizers
def influence_plot(influence_type_ = 'Dense', epochs = 200, _lambda = 0.3):
    dense_addn, counter, l2_need = 0, 0, 0
    for n in range(1, 6):
        model = Sequential()
        # 增加一层网络
        model.add(
            layers.Dense(8, input_dim=2, activation='relu')
        )
        # 9.8.2 网络层数的影响
        ## 如果是看 网络层影响 就分别看一下 增加 0-5层隐含层的效果
        if influence_type_ == 'Dense':
            print(f'查看网络{n}层数的影响')
            counter = n
            dense_addn = n  
            title_ = 'fc'
        # 9.8.3 Dropout 层的网络
        elif influence_type_ == 'Dropout':
            print(f'查看Dropout {n}层数的影响')
            counter = 0 
            dense_addn = 5
            title_ = 'drop'
        elif influence_type_ == 'l2':
            print(f'查看l2 :{_lambda}的影响')
            counter = n
            dense_addn = 3
            title_ = 'l2'
        else:
            print('influence_type_ in [Dense, Dropout, l2]')
            break

        for _ in range(dense_addn):
            # 9.8.4 正则化l2的影响
            if influence_type_ == 'l2':
                model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
            elif influence_type_ == 'Dropout':
                model.add(layers.Dense(64, activation='relu'))
            else:
                model.add(layers.Dense(32, activation='relu'))
            if counter < n:
                counter += 1
                model.add(layers.Dropout(rate=0.5))

        # 输出层
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss = 'binary_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy']
        )
        history = model.fit(x_tr, y_tr, epochs=epochs, verbose=0)
        acc_f = round(max(history.history['accuracy']), 2)
        print('最终acc:',acc_f)
        
        if influence_type_ == 'l2':
            title = f"{title_}-({_lambda}), acc:{acc_f:.2f}" 
        else:
            title = f"{title_}-({n})-[in-{n}{title_}s-out], acc:{acc_f:.2f}" 
        xx1, xx2 = pre_border_dt()
        preds = model.predict_classes(np.c_[xx1.ravel(), xx2.ravel()])
        print(f'finished {title}')
        make_plot(x_tr, y_tr, title, xx1, xx2, preds)
        if influence_type_ == 'l2':
            # 只循环一次
            break

    plt.show()

influence_plot('Dense', epochs = 500)
influence_plot('Dropout', epochs = 200)
for i in [0.1, 0.3, 0.8, 0.9]:
    influence_plot('l2', epochs = 500, _lambda = i)


