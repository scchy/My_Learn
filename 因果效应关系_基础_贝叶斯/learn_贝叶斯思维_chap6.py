# -*- coding: utf-8 -*-
#! /user/bin/python 3
# Author: Scc_hy
# Date:   2020-04-18
# Theme:  贝叶斯思维：统计建模的python学习法


# ====== 目录 ========
# &6 决策分析
#   - 6.4 PDF 的表示
#   - 6.5 选手的建模
# ====================
from scc_function.scc_bayes.scc_bayes import Pmf, Suite

#  =================================================================
#                     第六章   决策分析
#  =================================================================
## 

# 6.4 PDF 的表示
# ------------------------------------------------
from scc_function.scc_bayes.scc_bayes import Pmf, EstimatedPdf
import numpy as np
import csv
import scipy.stats as sts 

def ReadData(fl):
    """
    返回 序列 (price1 price2 bid1 bid2 diff1 diff2) 
    """
    f = open(fl)
    f_con = csv.reader(f)
    res = []
    for line in f_con:
        _head = line[0]
        data = line[1:]
        try:
            data = [int(x) for x in data]
            res.append(data)
        except:
            pass
    f.close()
    return list(zip(*res))


fil_name = r'C:\Users\29629\Desktop\ThinkBayes-master\code\showcases.2011.csv'
prices = ReadData(fil_name)

pdf = EstimatedPdf(prices)
low, high = 0, 75000
n = 101
xs = np.linspace(low, high, n)
pmf = pdf.MakePmf(xs.T)
pmf.d
pdf.kde(xs)

a = sts.gaussian_kde(xs)
a.evaluate(xs)


# 6.5 选手的建模
# ------------------------------------------------
