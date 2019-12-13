# python 3.6
# Author:              Scc_hy
# Create date:         2019-12-12
# Function:            统计学习方法
# Version :


import sys, os
from random import random
from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.model_selection import train_test_split
import pydotplus

def create_data(return_numpy=True):
    """
    载入iris的前一百的数据
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [i.replace(' (cm)', '') for i in df.columns]
    if return_numpy:
        data = np.array(df.iloc[:100, :])
        out_1, out_2 = data[:, :-1], data[:, -1]
    else:
        out_1, out_2 = df.iloc[:100, :-1], df.iloc[:100, -1]
    return out_1, out_2

# ================================================================
#                   第六章   Logistic
## 条件概率分布表示的分类模型
# ================================================================
## 6.1 
## -------------------------------------

x, y = create_data(True)
x_tr, x_te, y_tr, y_te = train_test_split(x[:,[0,1]], y, test_size = 0.3)


class Lg_reg_clf():
    def __init__(self, max_iter = 200, lr = 0.01):
        self.max_iter = max_iter
        self.lr = lr
        
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def data_matrix(self, x):
        one_x = np.ones((x.shape[0], 1))
        return np.c_[one_x, x]

    def fit(self, x, y):
        dt = self.data_matrix(x)
        m, n = dt.shape
        self.weights = np.zeros((n, 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(m):
                result = self.sigmoid(np.dot(dt[i], self.weights))
                error = y[i] - result
                self.weights += self.lr * error * np.transpose([dt[i]])
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(
            self.lr, self.max_iter))

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)


lr_clf = Lg_reg_clf()
lr_clf.fit(x_tr, y_tr)
lr_clf.score(x_te, y_te)


x_points = np.arange(4, 8)
x_points2 = -(lr_clf.weights[1] * x_points + lr_clf.weights[0])/ lr_clf.weights[2]

plt.plot(x_points, x_points2)
plt.scatter(x[:50, 0], x[:50, 1], label='0')
plt.scatter(x[50:, 0], x[50:, 1], label='1')
plt.legend()
plt.show()

## 6.2 sklearn 实例
## -------------------------------------
"""
solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是：
a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
"""
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=200)
clf.fit(x_tr, y_tr)
clf.score(x_te, y_te)

x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(x[:50, 0], x[:50, 1], 'bo', color='blue', label='0')
plt.plot(x[50:, 0], x[50:, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

