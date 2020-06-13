# python 3.6
# create date: 2020-06-13


# PCA实战
import numpy as  np

pric_ = np.array([10, 2, 1, 7, 3]).reshape((-1,1))
area_ = np.array([9, 3, 2, 6.5, 2.5]).reshape((-1,1))
array_ = np.c_[pric_, area_]
# 去中心化
array_sub = array_  - array_.mean(axis=0)
 
# 协方差矩阵
"""
[[var(x) cov(x, y)],
[cov(x, y), var(y)]]
"""
Q = np.array([[np.var(array_sub[:,0]), np.mean(array_sub[:,0]*array_sub[:,1]) ],
        [np.mean(array_sub[:,0]*array_sub[:,1]) , np.var(array_sub[:,1])]])
# 奇异分解
dig, U = np.linalg.eig(Q)
dig_arr = np.diag(dig)
# 主元素e1 奇异值对应的奇异向量
"""
e1 = np.dot(U, np.array([1, 0]))
e2 = np.dot(U, np.array([0, 1]))
e1, e2
T = np.c_[e1, e2]
np.dot(array_, T)
"""
np.dot(array_, U)

