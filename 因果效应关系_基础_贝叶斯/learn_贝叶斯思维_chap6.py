# -*- coding: utf-8 -*-
#! /user/bin/python 3
# Author: Scc_hy
# Date:   2020-04-18
# 2020-07-04
# 有点拖了
# Theme:  贝叶斯思维：统计建模的python学习法


# ====== 目录 ========
# &6 决策分析
#   - 6.4 PDF 的表示
#   - 6.5 选手的建模
#   - 6.6 似然度
#   - 6.7 更新
#   - 6.8 最优出价
#   - 6.9 讨论

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


# 6.5 选手建模
# ------------------------------------------------
"""
1- 怎么看待数据以及如何量化数据？
2- 怎么计算似然函数？ 即，对每个价格给出什么样的假设值，怎么计算数值的
条件似然度？

将选手作为一种误差特征已知的价格猜测仪来建模。欢聚话说。
当选手看到每组展品猜单个奖品的价格时——不考虑奖品是展品一部分这一事实(也就是不考虑总量)
——再把这些价格加起来，得到的综合为猜测价格guess
"""

"""
这一假设下，我们必须要回答的问题是"如果实际价格是price, 选手的估计价格就是猜测价格guessd 似然度？"

定义：
    error = price - guess 
问题： 选手的估计价格背离猜测误差error的似然度是什么？

diff = price - bid (奖品价格 - 选手出价)

假设 error 的分布是一个 和 diff方差相同，均值为0的高斯分布
"""
class Player():
    def __init__(self, player, bids, diffs):
        """
        cdf_diff 是出价差的累积分布
        pdf_error 是猜测误差分布的PDF 
        """
        self.pdf_price = EstimatedPdf(prices)
        self.cdf_diff = MakeCdfFromlist(diffs, name='')
        mean_ = 0
        std_ =  np.std(diffs)
        self.pdf_error = GaussianPdf(mean_, std_)

    def ErrorDensity(self, error):
        """
        通过给定值的错误来评估pdf_error， 结果是概率密度。
        """
        return self.pdf_error.Density(error)

# 6.6 似然度
# ------------------------------------------------
bids_lst1 = [i[2] for i in prices]
diffs_lst1 = [i[4] for i in prices]

from scc_function.scc_bayes.scc_bayes import Suite
class Price(Suite):
    def __init__(self, pmf, player):
        Suite.__init__(self, pmf)
        self.player = player
    
    def Likelihood(self, data, hypo):
        """
        Likelihood不需要计算概率；它只需要计算比例，
        只要左右likelihood的比例系数相同的，我们对后验分布进行
        归一化后就没问题了。
        所以概率密度时一个相当好的似然度方法
        """
        price = hypo
        guess = data

        error = price - guess
        like = self.player.ErrorDensity(error)
        return like
    
# 6.7 更新
# ------------------------------------------------
"""
Player提供了一个方法以选手的猜测来计算后验分布

def MakeBeliefs(self, guess):
    pmf = self.PmfPrice()
    self.prior = Price(pmf, self)
    self.posterior = self.prior.Copy()
    self.posterior.Update(guess)

PmfPrice 生成PDF的离散近似价格，我们用其构建先验概率。
PmfPrice 使用MakePmf，评估pdf_price序列的值。
n = 101 #
price_xs = np.linspace(0, 75000, n)

def PmfPrice(self):
    return self.pdf_price.MakePmf(self.price_xs)

结合连个信息源，过去展品的历史数据 和 你看到奖品后作出的猜测

我们把之前的历史数据当作先验概率和然后基于你的猜测去修正它。但同样的，
我们也可以用你的猜测作为先验而把基于历史的数据进行修正(译注：修正和更新都是Update)

或者你可以这么思考——最有可能的展品价格并不是你最初的猜测值——于是这一点也就没那么奇怪了 。

"""

# 6.8 最优出价
# ------------------------------------------------
"""
我们有一个后验分布，我们可以使用它来计算最优报价，我定义为预期收益最大化的报价

我将在本节中采用自顶向下的方法，这意味着武将先演示怎么使用， 再演示为什么如此。

GainCalculator提供ExcetedGains 为每次出价计算投标序列和预期收益

def ExpectedGains(self, low=0, high=75000, n=101):
    bids = np.linspace(low, high, n)
    gains = [self.ExpectedGains(bid) for bid in bids]
    return bids, gains

ExpectedGains 调用ExpectedGain计算对于一个给定的报价的预期值：
def ExpectedGain(self, bid):
    suite = self.player.posterior
    total = 0
    for price, prob in sorted(suite.Items()):
        gain = self.Gain(bid, price)
        total += prob * gain
    
    return total

ExpectedGain 遍历后验概率的值，给定展品的实际价格后计算每次出价的回报。它针对概率进行加权计算然后返回
总和。
ExpectedGain 调用Gain Gain通过报价和实际价格返回预期收益。

def Gain(self, bid, price):
    if bid > price:
        return 0
    diff = price - bid
    prob = self.ProbWin(diff)

    if diff <= 250:
        return 2 * price * prob
    else:
        return price * prob
    
如果你出价高了将衣物所获。反过来，我们计算出价和价格的差，这个决定了你获胜的概率。
如果差异小于250美元你就赢了。为了简单起见，我假设展品有相同的价格。因为这个结果是罕见的， 造成的
差别不大
"""
"""
最后我们要基于diff计算的赢的概率
def ProbWin(self, diff):
    prob = (self.opponent.ProbOverbid() + self.opponent.ProbWorseThan(diff))
    return prob 

如果你的对手出价高，你赢。否则的话， 你必须希望你的对手的出价差大于这个diff值，
Player提供了一些方法来计算着两个可能性：

def ProbOverbid(self):
    return self.cdf_diff.Prob(-1)

def ProbWorseThan(self, diff):
    return 1 - self.cdf_diff.Prob(diff)

这个计算过程是以对手的角度进行的， 对手计算的正是， 我出高的可能性 和 我出价超出diff的概率是对手

两个答案都是基于 diff的cdf ,如果对手的差异小于或等于1， 你赢。如果对手的diff比你大，你赢。否则你输

最后的计算最优报价的脚本：
def OptimalBid(self, guess, opponent):
    self.MakeBeliefs(guess)
    calc = GainCalculator(self, opponent)
    bids, gains = calc.ExpectedGains()
    gian, bid = max(zip(gains, bids))
    return bid, gain


"""

# 6.9 讨论
# ------------------------------------------------
"""
贝叶斯估计的特点之一就是结果来自后验分布这种形式。 经典的估计通常会生成一个单一的点估计或置信区间，
如果估计就是过程的最后一步，这就足够。但是如果你想以一个估计作为后续分析的输入，点估计和间隔往往没有多少帮助。

在这个例子中，我们使用后延分布来计算最优报价。 给定出价的回报是不对称和不连续的，所以单纯分析很难解决这个问题。
但用数值计算的方法就相对简单。

在你需要将后验概率带入后续分析而进行模型决策时，贝叶斯方法就相当有用了， 就如我们在本章做的一样。 另外，进行预测时，
贝叶斯方法也很有用
"""
