# python 3.6
# Author:              Scc_hy
# Create date:         2020-05-18
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
#                   第七章   SVM
# ================================================================
## 7.1 
## -------------------------------------






