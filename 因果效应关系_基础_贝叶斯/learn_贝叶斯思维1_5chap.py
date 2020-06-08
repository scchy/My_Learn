# -*- coding: utf-8 -*-
#! /user/bin/python 3
# Author: Scc_hy
# Date:   2020-04-18
# Theme:  贝叶斯思维：统计建模的python学习法


# ====== 目录 ========
# &1 贝叶斯定理
# &2 统计计算
# &3 估计
# &4 估计进阶
# &5 胜率
#  -- 5.4 加数
#  -- 5.5 最大值
# ====================
from scc_function.scc_bayes.scc_bayes import Pmf, Suite


#  =================================================================
#                     第一章   贝叶斯定理
#  =================================================================
## 1-3 曲奇饼问题
### 假设有两碗曲奇饼，碗1包含30个香草曲奇饼 和 10 个巧克力曲奇饼。碗2包含以上两种饼干各20个。
# 现在设想你在不看的情况下随机挑一碗那一个饼干，得到一块香草曲奇。我们的问题是从碗1拿到香草曲奇的概率？

# 简单算法
p_final = 30 / 50 
print('P(碗1|香草) = ', p_final)
# 贝叶斯条件概率
p1_xc = 30 / 40 
p2_xc = 0.5 
p_xc = 0.5 * p1_xc + 0.5 * p2_xc
p_1_xc =  0.5 * p1_xc / p_xc
print('P(碗1|香草) = ', p_1_xc)

# 1-6 M&M 豆问题
### 1995年 颜色搭配
M_1995 = {
    '褐色': 0.3,
    '黄色': 0.2,
    '红色': 0.2,
    '绿色': 0.1,
    '橙色': 0.1,
    '黄褐色': 0.1
}
M_1996 = {
    '蓝色': 0.24,
    '绿色': 0.2,
    '红色': 0.13,
    '黄色': 0.14,
    '橙色': 0.16,
    '黄褐色': 0.13
}
# 一袋1994，一袋1996. 从每个袋子里各取一个 M&M 豆。一个黄色一个绿色
# 求P(1994|黄色，1996_绿)
## p(黄色) 与 p(绿色) 独立
p_yellow = 0.5*(M_1996['黄色'] * M_1995['绿色'])+\
     0.5*(M_1995['黄色'] * M_1996['绿色'])
p_1994_yellow = 0.5*(M_1995['黄色'] * M_1996['绿色']) / p_yellow
print('P(1994|黄色，1996_绿) = ', p_1994_yellow)


# 1-7 Monty Hall难题
### 如果你参加节目，规则是这样：
# monty向你示意三个关闭的大门，然后告诉你每个门后都有一个奖品：
## 一个人奖品是一辆车，另外两个是像花生酱和假指甲这些小礼物
## 1- 游戏的目的时猜哪个门后有车。如果你才对了就可以拿走汽车
## 2- 你挑选一个门(A)
## 3- 在打开你选中的门前，为了增加悬念，momty会打开B或者C中没有车的门
### 来增加悬念
## 4- 然后Monty会给一个选择换还是不换

## 【暴力分析】：用你的一扇门 换 Monty的两扇门 ，显然合算
## 【概率分析】：
# A- 所选门后面是汽车
# B- 换门后后面是汽车
# C- Monty打开一扇空门  
# (1) 当你在考虑不换门得到汽车，其后是汽车的概率的是典型的贝叶斯问题
# 显然是求P(A|C)
# p_a = P(A) = 1/3
p_a = 1/3
## P(C) = 1/2 
## P(C|A) = P(C) = 1/2 
## P(C|A)*P(A) / (P(C|A)*P(A) + P(C|A_)*P(A_))
p_aa = p_a*1/2 / (p_a * 1/2 + 2/3 * 1/2)
# (2) 当你考虑换门得到汽车的时候
## P(C) = P(C|B) = 1 开一扇门已经发生,且只能开其中一个
## 就存在两种情况，每一种等得到的概率都是1 
p_b =  1 * p_a + 1 * p_a


#  =================================================================
#                     第二章   统计计算
#  =================================================================
# 2.1 分布
# ------------------------------------------------
from scc_function.scc_bayes.scc_bayes import Pmf
## 骰子的概率
pmf = Pmf()
for x in range(1,7):
    pmf.Set(x, 1/6)

pmf.d
## 单词频次
pmf = Pmf()
for i in range(ord('A'), ord('Z') + 1):
    pmf.Set(chr(i), 1)
## 概率标准化 使得之和为1
pmf.Normalize()
pmf.d
# 2.2 曲奇饼问题
### 假设有两碗曲奇饼，碗1包含30个香草曲奇饼 和 10 个巧克力曲奇饼。碗2包含以上两种饼干各20个。
# 现在设想你在不看的情况下随机挑一碗那一个饼干，得到一块香草曲奇。我们的问题是从碗1拿到香草曲奇的概率？
# ------------------------------------------------
pmf = Pmf()
pmf.Set('Bowl1', 0.5)
pmf.Set('Bowl2', 0.5)
pmf.Mult('Bowl1', 3/4) # 碗1中抽 香草曲奇
pmf.Mult('Bowl2', 1/2) # 碗2中抽 香草曲奇
# 进行归一化 就可以求出 
pmf.Normalize()
pmf.d


# 2.3 贝叶斯框架
class Cookies(Pmf):
    def __init__(self, hypos, mixes = None):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()
        self.mixes = {
            'Bowl1': {'vanilla':0.75, 'chocolate':0.25},
            'Bowl2': {'vanilla':0.5, 'chocolate':0.5}
        } if mixes is None else mixes
    
    def Update(self, data):
        """
        更新每个假设下，该情况的概率 * 假设的概率
        """
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo , like)
        self.Normalize()
    
    def Likelihood(self, data, hypo):
        """
        某个特征在某个假设中的概率
        'Bowl1': {'vanilla':0.75, 'chocolate':0.25}
        如:
            # vanilla 在假设Bowl1 中的概率
            self.Likelihood('vanilla', 'Bowl1')
        """
        mix = self.mixes[hypo]
        return mix[data]


hypos = ['Bowl1', 'Bowl2']
pmf = Cookies(hypos)
pmf.Update('vanilla')
pmf.d


# 2.4 Monty Hall难题
class Monty(Pmf):
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo , like)
        self.Normalize()
    
    def Likelihood(self, data, hypo):
        if data == hypo:
            return 0
        elif hypo == 'A':
            return 0.5
        else:
            return 1

hypos = 'ABC'
mty = Monty(hypos)
mty.Update('B')
mty.d

# 2.5 封装框架
class Suit(Pmf):
    def __init__(self, hypo=tuple()):
        """
        初始化分配
        """
    def Update(self, data):
        """
        更新基于该数据的假设
        """
    
    def Print(self):
        """
        打印出假设和她们的概率
        """


class Monty(Suite):
    def Likelihood(self, data, hypo):
        if data == hypo:
            return 0
        elif hypo == 'A':
            return 0.5
        else:
            return 1

s = Monty('ABC')
s.Update('B')
s.Print()



# 2.6 M&M豆问题
M_1995 = {
    '褐色': 0.3,
    '黄色': 0.2,
    '红色': 0.2,
    '绿色': 0.1,
    '橙色': 0.1,
    '黄褐色': 0.1
}
M_1996 = {
    '蓝色': 0.24,
    '绿色': 0.2,
    '红色': 0.13,
    '黄色': 0.14,
    '橙色': 0.16,
    '黄褐色': 0.13
}

# 封装假设
hypoA = {'bag1': M_1995, 'bag2': M_1996}
hypoB = {'bag2': M_1995, 'bag1': M_1996}
hypo_dct = {'A': hypoA, 'B': hypoB}
class M_M(Suite):
    def dct_in(self, hypo_dct):
        self.hypo_dct = hypo_dct

    def Likelihood(self, data, hypo):
        bag, color = data
        mix = self.hypo_dct[hypo][bag] 
        like = mix[color]
        return like # 找到事件的概率

s = M_M('AB')
s.dct_in(hypo_dct)
s.Update(('bag1', '黄色'))
s.Update(('bag2', '绿色'))
s.Print()



#  =================================================================
#                     第三章   估计
#  =================================================================

# 3.1 骰子问题
# ------------------------------------------------
class Dice(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1.0/hypo



suite = Dice([4, 6, 8, 12, 20])
suite.d
suite.Update(6)

for roll in [6, 8, 7,7 ,5, 4]:
    suite.Update(roll)
suite.d

# 3.2 火车头问题
# ------------------------------------------------
"""
铁路上以1到N命名火车头。有一天你看到一个标号60的火车头，请估计铁路上
有多少火车头。
"""
hypos = list(range(1,1001))

class Train(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1.0/hypo


def Mean(suite):
    total = 0 
    for hypo, prob in suite.Items():
        total += hypo * prob
    return total

suite = Train(hypos)
suite.Update(60)
suite.Mean()
print(Mean(suite))

# 使用后验概率的平均值来作为估计值会减少从长远看的均方差

# 3.3 怎样看待先验概率
# ------------------------------------------------

for data in [60, 30, 90]:
    suite.Update(data)

suite.Mean()
print(Mean(suite))

# 3.4 其他先验概率
# ------------------------------------------------
"""
公司规模的分布往往遵循幂律函数
幂律函数
PMF(x) = (1/x)^a
a 为一个通常接近于1的参数

"""
class Train(Dice):

    def __init__(self, hypos, alpha = 1.0):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, hypo ** (-alpha))
        self.Normalize()

hypos = list(range(1,1001))
suite = Train(hypos)

for data in [60, 30, 90]:
    suite.Update(data)

# suite.Mean()
print(Mean(suite))


# 3.5 置信区间
# ------------------------------------------------
"""
计算置信区间的一个简单方法是在后验概率分布中累加其中的概率，
并记录对应于概率5%和95%的值。 
也就是说，第5 和 第95 百分位

"""

def Percentile(Pmf, percentage):
    p = percentage / 100.0
    total = 0
    for val, prob in Pmf.Items():
        total += prob
        if total >= p:
            return val



Percentile(suite, 5), Percentile(suite, 95)

# 3.6 累积分布函数
# ------------------------------------------------

cdf = suite.MakeCdf()
cdf.Percentile(5), cdf.Percentile(95)



#  =================================================================
#                     第四章   估计进阶
#  =================================================================
# 4.1 欧元问题
# ------------------------------------------------
"""
假设 x 从0-101 x/100 为当前假设下 出现正面的概率
如果硬币完全均匀，那个么这个x应该是接近50% 即该假设的后验概率最大
"""
class Euro(Suite):
    def Likelihood(self, data, hypo):
        x = hypo
        if data == 'H':
            return x/100
        else:
            return 1-x/100

    def UpdateSet(self, dataset):
        for d in dataset:
            for hypo in self.Values():
                like = self.Likelihood(d, hypo)
                self.Mult(hypo, like)
        return self.Normalize()

suite = Euro(range(101))
dt = 'H'*140 + 'T'*110

for d in dt:
    suite.Update(d)

import matplotlib.pyplot as plt 
plt.plot(list(range(101)), [i[1] for i in suite.Items()])
plt.show()


# 4.2 后验概率的概述
# ------------------------------------------------
## 1- 取最大
suite.MaximumLikelihood()
## 2- 取均值
suite.Mean()
## 3- 取中位数
Percentile(suite, 50)
## 4- 置信区间
Percentile(suite, 5), Percentile(suite, 90)



# 4.3 先验概率的淹没
# ------------------------------------------------
"""
如果硬币是偏心的，可以相信X会大幅偏离50%,但如果偏心
到使得x是10% 90%就不大可能了。
更合理的选择是在50%附近有较高概率，而在那些极端（10%, 90%）概率
较低的一个先验。
"""
def TrianglePrior():
    suite = Euro()
    for x in range(51):
        suite.Set(x, x)
    for x in range(51,100):
        suite.Set(x, 100-x)
    suite.Normalize()

# 4.4 优化 
# ------------------------------------------------
## 不用每次都做 Normalize()

# def UpdateSet(self, dataset):
#     for d in dataset:
#         for hypo in self.Values():
#             like = self.Likelihood(d, hypo)
#             self.Mult(hypo, like)
#     return self.Normalize()

suite.UpdateSet(dt)
import matplotlib.pyplot as plt 
plt.plot(list(range(101)), [i[1] for i in suite.Items()])
plt.show()


# 4.5 Beta分布
# 先将函数敲上
# ------------------------------------------------
"""
beta分布定义在0到1的区间上， 所以它是一个描述比例和概率的自然选择

alpha=1 beta=1 就是从0到1的均匀分布
"""





#  =================================================================
#                     第五章   胜率
#  =================================================================
# 5.1 胜率
# ------------------------------------------------
"""
当概率较低，通常称为赔率(odds against)，而不是胜率(odds in favor)
例如， 如果我的马只有10%获胜的机会，我会说赔率是9:1。

"""
def Odds(p):
    return p /(1-p)

def Probability(o):
    return o / (o + 1)



# 5.2 贝叶斯定理的胜率形式
# ------------------------------------------------
"""
P(H|D) = [P(H)P(D|H) / P(D)]

P(A|D) / P(B|D) = P(A)P(D|A) / [P(B)P(D|B)]

# 如果A B互斥时 p(B) = 1 - p(A)
o(A|D) = o(A)[ p(D|A) / p(D|B) ]

在字面形式上，这说明了后验赔率是先验赔率乘以似然比，

这种形式利于在脑中计算贝叶斯概率
"""
"""
两婉曲奇饼。婉1 包含30个香草， 10个巧克力。
婉2， 20个香草，20个巧克力。

随机选择一个，然后是香草  来自婉1的概率

先验概率是 50% ， 所以胜率是1:1  o(A) = 1
似然度是 3/4 / 1/2 = 3/2  o(A|D) = 3:2

所以后验概率就是 3/2
对应的概率就是 3/(3+2)


"""
3/4 * 0.5 / (0.5*( 3/4 + 1/2 ))




# 5.3 奥粒弗的血迹
# ------------------------------------------------
"""
在一个犯罪现场， 有两人遗留了血迹。一名疑犯奥利弗经过测试发现是'O'型血。而发现
的痕迹中血型分别是'O'型(一种本地人口的常见血型，有60%的概率)和'AB'型(一种罕见的血型，
概率为1%)，那么这些数据是否支持奥利弗是疑犯之一？

如果奥利弗是犯罪现场留下血迹的人之一，就解释了那个'O'血型证据样本的由来，因此数据的概率
就是在人群中随意挑中一个‘AB’血型的概率 1%

如果不是，则是随机抽取的组合 O-AB AB-O 2 * 0.6*0.01 = 1.2%.
该情况的数据的似然度还要稍微高一些，所以血液证据并不能证明奥利弗的犯罪嫌疑。
"""
"""
即，该数据由一个常见事件——'O'型血，和一个罕见事件——'AB'型血构成。如果奥利弗与常规事件相关
，这使得罕见的事件还是无法解释。如果奥利弗与常规事件无关，那么我们有2中可能找到
'AB'型血的疑犯。两种情况中的这一因素导致了差异
"""

from scc_function.scc_bayes.scc_bayes import Pmf, Suite, SampleSum



# 5.4 加数
# ------------------------------------------------
class Die(Pmf):
    def __init__(self, sides):
        Pmf.__init__(self)
        for x in range(1, sides + 1):
            self.Set(x, 1)
        self.Normalize()

# 创建6面的骰子
d6 = Die(6)
dice = [d6] * 3
# 产生1000次转动3个骰子的样本
# 随机选取对象 汇总，查看汇总的值出现的概率
three = SampleSum(dice, 1000)
three.name = 'sample'
three.Print()


import matplotlib.pyplot as plt
v_lst = sorted(three.Items(), key=lambda pair:pair[0])
plt.plot([i[0] for i in v_lst], [i[1] for i in v_lst])
plt.show()


# 5.5 最大化
# ------------------------------------------------
"""
三种方法来计算一个最大值的分布
模拟：
    给定一个Pmf, 代表单一选择的分布，可以生成随机样本。找到最大值和模拟最大值的累积分布。

枚举：
    给定两个Pmf, 可以枚举所有可能的数值对， 并计算分布的最大值

指数计算：
    如果我们将一个Pmf转换成Cdf，有一个简单而有效的算法查找最大Cdf. 模拟最大值的代码与模拟求和的代码几乎相同

Pmf.Max 与  cdf.Max实现是一样的

cdf(5) 表示从分布中随机选取一个值小于等于5的概率

从cdf1中取出x, 从cdf2中取出y， 计算做大值Z=max(x, y),则Z小于或等于5的可能性是多少
如果x y 是独立分布的
那个 cdf3(5) = cdf1(5)*cdf2(5)
从一个分布中选择k次，
cdfk(z)=cdf1(z)^k

求最大值就是 每个累计概率的 k方

"""

from scc_function.scc_bayes.scc_bayes import Pmf, Suite, SampleSum
six_tz = Pmf(dict(zip(range(1,7), [1]*6)))
"""
out_pmf = Pmf()
for k1, v1 in self.Items():
    for k_o, v_o in otherPmf.Items():
        out_pmf.Incr(k1+k_o, v1*v_o)
return out_pmf
属性 __add__ 
"""
three_exact = six_tz +  six_tz + six_tz
three_exact.name = 'exact'
three_exact.Print()

best_attr_cdf = three_exact.Max(6)
best_attr_pmf = best_attr_cdf.MakePmf()


import matplotlib.pyplot as plt
v_lst = sorted(best_attr_pmf.Items(), key=lambda pair:pair[0])
plt.plot([i[0] for i in v_lst], [i[1] for i in v_lst])
plt.show()

