# python 3.6
# Author:              Scc_hy
# Create date:         2019-11-15
# Function:            统计学习方法
# Version :
""" 示例
# ================================================================
#                   第一章   统计学习方法概论
# ================================================================
## 1.1 使用最小二乘法拟和曲线
## -------------------------------

### 1.1.1 xxx


第三章 kd tree search 后面需要再看一次 2019-11-26
""""
import sys, os
from random import random
from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import leastsq
"""leastsq
leastsq(func,   #  误差函数
        x0,     #  The starting estimate for the minimization 表示函数的参数
        args=(), # Any extra arguments to func are placed in this tuple
        Dfun=None,
        full_output=0,
        col_deriv=0,
        ftol=1.49012e-08,
        xtol=1.49012e-08,
        gtol=0.0,
        maxfev=0,
        epsfcn=0.0,
        factor=100,
        diag=None,
        warning=True)
"""
from sklearn.datasets import load_iris
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import train_test_split
from collections import Counter # 将列表都是数，读取成字典
import pprint
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
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
#                   第一章   统计学习方法概论
# 进行模型选择或者说提高学习的【泛化能力】是一个重要问题
# 模型选择的方法有正则化与交叉验证
# ================================================================

## 1.1 使用最小二乘法拟和曲线
## -------------------------------
### 举例 用目标函数 + 正态分布的噪音干扰 ， 用多项式去拟合

# 目标函数
def real_func(x):
    return np.sin(2 * np.pi * x)

# 拟合多项式
def fit_func(p, x):
    """
    math:`x^2 + 2x + 3`, whereas `poly1d([1, 2, 3])
    """
    f_x = np.poly1d(p)
    return f_x(x)

# 残差
def res_func(p, x, y):
    return fit_func(p, x) - y


# 十个点
x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
# 加上正态分布噪音的目标函数的值
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]

np.random.random(2+1)


def fitting(res_func, M = 0):
    # 随机初始化多项式参数
    p_init = np.random.random(M+1)
    # 最小二乘法
    p_lsq = leastsq(res_func, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq[0])
    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve') # 取最佳参数拟合
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq


# M=0
p_lsq_0 = fitting(M=0, res_func=res_func)  # y=ax
p_lsq_0 = fitting(M=1, res_func=res_func)   # y=ax + b
p_lsq_0 = fitting(M=2, res_func=res_func)   # y=ax2 + bx + c
p_lsq_0 = fitting(M=3, res_func=res_func)   # y=ax3 + bx2 + cx + d

p_lsq_9 = fitting(M=9, res_func=res_func)

## 1.2 正则化
"""
回归问题中，损失函数是平方损失，正则化可以是参数向量的L2范数, 也可以是L1范数。
L1: regularization*abs(p)
L2: 0.5 * regularization * np.square(p)
"""
## -------------------------------
reg_num = 0.0001

def res_fun_reg(p, x, y):
    ret = res_func(p, x, y)
    ret = np.append(ret,
                    np.sqrt(0.5 * reg_num * np.square(p))) # L2
    return ret

help(np.append)

p_lsq_9 = fitting(M=9, res_func=res_fun_reg)


# ================================================================
#                   第二章   感知机
# 0- 感知机是根据输入实例的特征向量  x  对其进行二类分类的线性分类模型
# 1- 感知机学习的策略是极小化损失函数 (损失函数对应于误分类点到分离超平面的总距离)
# 2- 感知机学习算法是基于随机梯度下降法的对损失函数的最优化算法，有原始形式和对偶形式
# 3- 当训练数据集线性可分时，感知机学习算法是收敛的
#(当训练数据集线性可分时，感知机学习算法存在无穷多个解，其解由于不同的初值或不同的迭代顺序而可能有所不同)
# ================================================================
## 2.1 二分类
## -------------------------------
# 0- load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = [i.replace(' (cm)', '') for i in df.columns]

## 查查看数据
sns.lmplot(x='sepal length', y='sepal width', hue= 'label', data = df
        , fit_reg=False, markers=["o", "x", "1"])
plt.show()

# 1- 取前一百的数据
data = df.iloc[:100, [0, 1, -1]].values
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y]) # 转为2分类

X.shape
# 2- Pereptron 感知机
## 数据线性可分，二分类数据
## 此处为一元一次线性方程
class My_perceptron(object):
    """Scc_hy
    随机梯度下降法 Stochastic Gradient Descent
    随机抽取一个误分类点使其梯度下降。
    $w = w + \eta y_{i}x_{i}$
    $b = b + \eta y_{i}$
    param learn_rate: learning rate
    param error_rate: 容忍错误率
    param check_freq: 每迭代几次检查一次
    param max_ite: 最大迭代次数
    """

    def __init__(self, learn_rate=0.1, error_rate = 0.01
                , check_freq = 10
                , max_iter = 100):
        self.b = 0
        self.learn_rate = learn_rate
        self.error_rate = error_rate
        self.check_freq = check_freq
        self.max_iter = max_iter


    def sign(self, x):
        y = np.sign(np.dot(x, self.w) + self.b)
        return y

    # 随机梯度下降
    def fit(self, x_tr, y_tr):
        m, n = x_tr.shape
        self.w = np.ones(n , dtype=np.float32)
        cnt = 0
        max_iter = self.max_iter
        while self.max_iter:
            cnt += 1
            for d in range(m):
                index_i = list(range(m))
                np.random.shuffle(index_i)
                x = x_tr[index_i][d]
                y = y_tr[index_i][d]
                if y * self.sign(x) <= 0: # 分类错误
                    self.w += self.learn_rate * np.dot(y, x)
                    self.b += self.learn_rate * y
            ## 每check_freq 检查一次
            if cnt % self.check_freq == 0:
                error_p = sum((y_tr * self.sign(x_tr) - 1) / -2)  # 错误的个数
                print("iter [{}]:..\terror_p: {}".format(
                    max_iter - self.max_iter, error_p))
                if error_p <= self.error_rate * m: ## 如果已经满足最大错误率则停止迭代
                    break
            self.max_iter -= 1
        return 'Perceptron Model'

    def predict(self, x_points ,class_ = False):
        if class_:
            out = np.sign(-(self.w[0] * x_points + self.b))
        else:
            out =  -(self.w[0] * x_points + self.b) / self.w[1]
        return out


perceptron = My_perceptron(max_iter=1000, error_rate=0.0001, check_freq=100)
perceptron.fit(X, y)
# X.shape[0] * 0.01

# 3- 预测画图
x_points = np.linspace(4, 7, 10)
"""
L(w, b) = y_i((w_1, w_2)^T . (x_1 , x_2)^T + b) = 0
x_2 = (w_1 * x_1 + b) / (w_2)
"""
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]


def plt_perceptron(x_points, pred_y):
    plt.plot(x_points, pred_y)
    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


plt_perceptron(x_points, y_)

## 2.2 sklearn 实例
## -------------------------------
import sklearn
from sklearn.linear_model import Perceptron


clf = Perceptron(fit_intercept=True,
                max_iter=1000,
                tol=None,
                shuffle=True)
clf.fit(X, y)
# 画感知机的线 (为什么这么画)
x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_points + clf.intercept_)/clf.coef_[0][1]
plt_perceptron(x_points, y_)


# ================================================================
#                   第三章   K近邻法
# 0- $k$近邻法的基本做法是：对给定的训练实例点和输入实例点，首先确定输入实例点的$k$个最近邻训练实例点，然后利用这$k$个训练实例点的类的多数来预测输入实例点的类
# 1- 常用的分类决策规则是多数表决，对应于经验风险最小化
# 2- 快速搜索k个最近邻点 kd树是二叉树，表示对$k$维空间的一个划分，其每个结点对应于$k$维空间划分中的一个超矩形区域。利用kd树可以省去对大部分数据点的搜索， 从而减少搜索的计算量。
# ================================================================
## 3.1 距离度量
## -------------------------------

def M_distance(x, y, p = 2):
    """
    param x: np.array
    param y: np.array
    """
    x_n, y_n = x.shape[0], y.shape[0]
    if x_n == y_n:
        sum_a = sum(np.power(np.abs(x - y), p))
    else:
        sum_a = 0
    return np.power(sum_a, 1 / p)

### 3.1.1 distance type
x1 = np.array([1, 1])
x2 = np.array([5, 1])
x3 = np.array([4, 4])
for i in range(1, 5):
    r = {'x1-{}'.format(c) : '{:.3f}'.format(M_distance(x1, c, p = i)) for c in [x2, x3]}
    print('The minist distance(typ_{}) with x1 is:  '.format(i),min(zip(r.values(), r.keys())))


## 3.2 iris K近邻
## -------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['label'] = iris.target
df.columns = [i.replace(' (cm)', '') for i in df.columns]

sns.lmplot(x='sepal length', y='sepal width',hue='label' ,data = df.loc[:100,:], fit_reg=False)
plt.show()


data = df.iloc[:100, [0, 1, -1]].values
X, y = data[:, :-1], data[:, -1]
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

class M_KNN(object):
    def __init__(self, n_neighbor = 3, p = 2):
        """
        param n_neighbor: int 临近点个数
        param p: distance type
        """
        self.n = n_neighbor
        self.p = p

    def fit(self, x_tr, y_tr):
        self.x_tr = x_tr
        self.y_tr = y_tr

    def predict_point(self, x):
        knn_list = []
        for i in range(self.n): # 构建n个临近的列表
            dist = np.linalg.norm(x - self.x_tr[i], ord = self.p) # 求范数
            knn_list.append((dist, self.y_tr[i]))

        for i in range(self.n, len(self.x_tr)): # 更新全集中距离最近的点
            max_index = knn_list.index(max(knn_list, key = lambda x:x[0]))
            dist = np.linalg.norm(x - self.x_tr[i], ord=self.p)  # 求范数
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_tr[i])
        # 统计
        knn_tp = [k[-1] for k in knn_list]
        count_dict = Counter(knn_tp)
        max_count_tp = sorted(count_dict.items(),
                        key=lambda x: x[1], reverse=True)[0][0]
        return  max_count_tp

    def predict(self, x_array):
        pred_y = np.ones((x_array.shape[0], 1))
        for i in range(x_array.shape[0]):
            pred_y[i] = self.predict_point(x_array[i])
        return pred_y.flatten()

    def score(self, x_te, y_te):
        pred_y = self.predict(x_te)
        return sum(pred_y == y_te) / len(y_te)

clf = M_KNN(n_neighbor = 3, p = 2)
clf.fit(x_tr, y_tr)
clf.predict_point(np.array([6, 3]))
clf.score(x_te, y_te)


# draw it
sns.lmplot(x='sepal length', y='sepal width', hue='label',
        data=df.loc[:100, :], fit_reg=False)
plt.scatter(6, 3 , color='red', edgecolors='grey')
plt.show()


### 3.2.1 iris K近邻 sklearn实例
from sklearn.neighbors import KNeighborsClassifier
clf_sk = KNeighborsClassifier(p=2)
clf_sk.fit(x_tr, y_tr)
clf_sk.score(x_te, y_te)


## 3.3 kd树
## -------------------------------
"""
kd树是一种对k维空间中的实例点进行存储以便对其进行快速检索的树形数据结构。
kd树是二叉树，表示对$k$维空间的一个划分（partition）。构造kd树相当于不断地用垂直于坐标轴的超平面将$k$维空间切分，构成一系列的k维超矩形区域。kd树的每个结点对应于一个$k$维超矩形区域。

依次选择坐标轴对空间切分，选择训练实例点在选定坐标轴上的中位数 （median）为切分点，这样得到的kd树是平衡的。
注意，平衡的kd树搜索时的效率未必是最优的

"""
### 构造平衡kd树算法
# kd-tree 每个结点中主要包含的数据结构如下：
class M_KdNode(object):
    def __init__(self, dom_elt, split_n, left, right):
        self.dom_elt = dom_elt # K维向量结点(K维空间中的一个样本点)
        self.split_n = split_n # 整数(进行分割维度的序号)
        self.left = left       # 该结点分割超平面左子空间构成的kd-tree
        self.right = right     # 该结点分割超平面右子空间构成的kd-tree

class M_KdTree():
    def __init__(self, data):
        """
        param data: list

        """
        self.k = len(data[0]) # 数据维度
        self.root = self.CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点

    def CreateNode(self, split, data_set ): # 按照split维划分数据集exset创建KdNode
        if not data_set: # 穷尽递归
            return None
        # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
        # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
        #data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
        else:
            data_set.sort(key=lambda x: x[split]) # 进行切分的维度split
            split_pos = len(data_set)  // 2
            median = data_set[split_pos]  # 切分维度的中位数
            split_next = (split + 1) % self.k  # 下一个切分的维度， 循环维度
            return M_KdNode(
                median,
                split,
                self.CreateNode(split_next, data_set[:split_pos]), ## 拆分左右 穷尽
                self.CreateNode(split_next, data_set[split_pos + 1:]),
            )


# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)


# 对构建好的KD树进行搜索，寻找与目标点最近的样本点：
from collections import namedtuple

# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple",
                    "nearest_point  nearest_dist  nodes_visited")
# kd-tree search
def find_nearest(tree, point):
    k = len(point)

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            # nearest_point  nearest_dist  nodes_visited
            return result([0] * k, float('inf'), 0)

        nodes_visited = 1
        s = kd_node.split_n  # 进行分割的维度
        pivot = kd_node.dom_elt # 进行分割的轴

        if target[s] <= pivot[s]: # 如果目标点第S维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left # 下一个访问节点为左子树根节点
            further_node = kd_node.right
        else:
            further_node, nearer_node = kd_node.left, kd_node.right

        temp1 = travel(nearer_node, target, max_dist)

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist # 最近点将以目标点为球心， max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])
        if max_dist < temp_dist: # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)
        ## 计算目标点与分割点的欧式距离
        temp_dist = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist: # 如果'更近'
            nearest = pivot  # 更新最近点
            dist = temp_dist # 更新最忌距离
            max_dist = dist  # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)
        if temp2.nearest_dist < dist :
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist   # 更新最忌距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float('inf')) # 从根节点开始递归


data =  [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
kd = M_KdTree(data)
preorder(kd.root)
# kd.root.left.left.dom_elt

# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]

# 产生n个k维随机向量

def random_points(k, n):
    return [random_point(k) for _ in range(n)]


ret = find_nearest(kd, [3, 4.5])
print(ret)



from time import clock
from random import random
N = 400000
t0 = clock()
kd2 = M_KdTree(random_points(3, N))            # 构建包含四十万个3维空间样本点的kd树
ret2 = find_nearest(kd2, [0.1, 0.5, 0.8])      # 四十万个样本点中寻找离目标最近的点
t1 = clock()
print("time: ", t1-t0, "s")
print(ret2)



# ================================================================
#                   第四章   朴素贝叶斯
# 0- 典型的生成学习方法
# 1- 强假设，高效，性能不一定高
# ================================================================
## 4.1 GaussianNB 高斯朴素贝叶斯
## -------------------------------------
### 0 加载数据
"""
@staticmethod 静态方法只是名义上归属类管理，但是不能使用类变量和实例变量，是类的工具包
放在函数前（该函数不传入self或者cls），所以不能访问类属性和实例属性
"""
x, y = create_data()
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.3)

x_te[0], y_te[0]

class M_NaiveBayes():
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod # 不能使用类变量和实例变量
    def mean(x):
        return sum(x) / len(x)

    # 标准差
    def stdev(self, x):
        avg = self.mean(x)
        return np.sqrt(sum([np.power(i - avg, 2) for i in x])) / len(x)

    # 概率密度函数
    def gaussion_prob(self, x, mean, stdev):
        exponent = np.exp(-np.power(x - mean, 2) / (2 * np.power(stdev, 2)) )
        return (1 / (np.sqrt(2 * np.pi ) * abs(stdev))) * exponent

    # 处理x_tr
    def summarize(self, train_data):
        # 获取列均值和std
        sum_d = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return sum_d

    # 分类别求出数学期望和标准差
    def fit(self, x, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(x, y):
            data[label].append(f) # 标签下所有的数据归类
        self.model = {
            # {0.0: [(5.0617647058823545, 0.05611633767777412), (3.4911764705882358, 0.05960766291783598), (1.4823529411764704, 0.02865816557105185), (0.25000000000000006, 0.020482923937035463)]
            # , 1.0: [(5.936111111111112, 0.08517889469527601), (2.7666666666666666, 0.055277079839256664), (4.26111111111111, 0.07052249638640598), (1.3194444444444442, 0.033522482884037134)]}
            label: self.summarize(value)
            for label, value in data.items()
        }
        return 'gaussianNB train done!'

    # 计算概率
    def caculate_prob(self, input_data):
        # 仅对一个实例进行计算
        prob = {}
        for label, value in self.model.items():
            prob[label] = 1
            for i in range(len(value)):  # 遍历每个特征的均值 标准差
                mean, stdev = value[i]
                # 计算一个实例的 每个特征的高斯概率
                prob[label] *= self.gaussion_prob(input_data[i], mean, stdev)
        return prob

    # 类别
    def predict(self, x_point):
        label = sorted(self.caculate_prob(x_point).items(),
        key = lambda x: x[-1])[-1][0]
        return label

    def score(self, x_te, y_te):
        right = 0
        for x, y in zip(x_te, y_te):
            label = self.predict(x)
            if label == y:
                right += 1
        return right / len(x_te)


model = M_NaiveBayes()
model.fit(x_tr, y_tr)
print(model.predict(np.array([4.4,  3.2,  1.3,  0.2])))
model.score(x_te, y_te)


## 4.2 skleran 实例
## -------------------------------------
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_tr, y_tr)
clf.score(x_te, y_te)
clf.predict([[4.4,  3.2,  1.3,  0.2]])

from skleran.naive_bayes import BernoulliNB, MultinomialNB # 伯努利模型和多项式模型


# ================================================================
#                   第五章   决策树
# ================================================================
## 5.1 书上题目5.1
## -------------------------------------
def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)

def calc_ent(dt):
    """
    计算目标的熵
    """
    dt_len = len(dt)
    unqs, inds = np.unique(dt[:, -1], return_counts=True)
    label_count = {label : value for label, value in zip(unqs, inds)}
    ent = -sum([(p / dt_len) * np.log2(p / dt_len)
                for p in label_count.values()])
    return ent

# 经验条件熵
def cond_ent(dt, col=0):
    """Scc_hy
    计算经验熵  
    param dt: numpy  
    param col: int 第一个特征  
    """
    dt_len = len(dt)
    unqs, inds = np.unique(dt[:, col], return_counts=True)
    every_ent_list = []
    for i, cnt in zip(unqs, inds):
        indices = dt[:, col] == i
        dt_tp = dt[:, [col, -1]][indices]
        # 计算当前特征子集下目标的熵
        ent_i = calc_ent(dt_tp)
        p_i = cnt / dt_len
        every_ent_list.append(ent_i * p_i)
    return sum(every_ent_list)

# 信息增益
def info_gain(ent, cond_ent):
    return ent - cond_ent



def info_gain_train(dt):
    feats = dt.shape[1] - 1
    ent_target = calc_ent(dt)
    info_gain_list = []
    nodes, nodes_infogian = 0, 0
    for i in range(feats):
        i_info_gain = info_gain(ent_target, cond_ent(dt, col=i))
        info_gain_list.append([i, i_info_gain])
        try:
            print('特征({})-info_gain: {:.3f}'.format(labels[i],i_info_gain))
        except:
            pass
        # 比较大小
        if nodes_infogian <= i_info_gain:
            nodes_infogian = i_info_gain
            nodes = i
    print('特征({})的信息增益最大，选择为根节点特征'.format(labels[nodes]))
    return (nodes, nodes_infogian)

np_dt = np.array(train_data)
cond_ent(np_dt, col = 0)
info_gain_train(np_dt)


## 5.2 利用ID3算法生成决策树
## -------------------------------------
class Node():
    def __init__(self, root = True, label = None, feature_name = None, feature = None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {
            'label': self.label,
            'feature': self.feature,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)




## 5.3 sklearn 实例
## -------------------------------------
def Tree_graph(decision_tree, feature_names, class_names,  file_root):
    """
    将决策树导出为PDF-到指定路径
    :param - decision_tree : skearn_clf
    :param - feature_names : list 
    :param - class_names : np.array
    :param - fil_root : file path
    """
    dot_data = export_graphviz(decision_tree, out_file=None,
                            feature_names=feature_names,
                            class_names=class_names, special_characters=True,
                            rounded=True, filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return graph.write_pdf(file_root)


x, y = create_data(False)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(x_tr, y_tr)
clf.score(x_te, y_te)
file_root = r'C:\Users\dell\Desktop\t.pdf'
Tree_graph(clf, list(x_tr.columns), np.array(['0','1']), file_root)



# mt = np.matrix([[1,1,-2],[1,0,-1],[-2,-1,2]])
# mt.I
