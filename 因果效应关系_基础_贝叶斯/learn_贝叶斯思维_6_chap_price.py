# -*- coding: utf-8 -*-
#! /user/bin/python 3
# Author: Scc_hy
# Date:   2020-07-04
# Theme:  第六章 价格

from __future__ import print_function

import csv
import numpy as np
import scc_function.scc_bayes as bys

import matplotlib.pyplot as plt

fil_name = r'C:\Users\29629\Desktop\ThinkBayes-master\code\showcases.2011.csv'
fil_name1 = r'C:\Users\29629\Desktop\ThinkBayes-master\code\showcases.2012.csv'
FORMAT = ['png', 'pdf', 'eps']

def ReadData(fil_name):
    f = open(fil_name)
    reader = csv.reader(f)
    res = []
    for t in reader:
        _heading = t[0]
        data = t[1:]
        try:
            data = [int(x) for x in data]
            res.append(data)
        except ValueError:
            pass
    f.close()
    return list(zip(*res))



class Price(bys.Suite):
    def __init__(self, pmf, player, name = ''):
        """
        pmf 
        player: Player object
        name: string
        """
        bys.Suite.__init__(self, pmf, name=name)
        self.player = player
    
    def Likelihood(self, data, hypo):
        """
        hypo: 实际价格
        data: 选手的猜测
        """
        price = hypo 
        guess = data 
        error = price - guess
        like = self.player.ErrorDensity(error)
        return like
    

class GainCalulator(object):
    def __init__(self, player, opponent):
        """
        player: 玩家
        opponnet: 对手玩家
        """     
        self.player = player
        self.opponnet = opponent
    
    def ExpectedGains(self, low=0, high=75000, n=101):
        """ 
        计算预期收益
        low: 最低价
        high: 最高价
        n: 样本数
        """
        bids = np.linspace(low, high, n)
        gains = [self.ExpectedGain(bid) for bid in bids]
        return bids, gains
    
    def ExpectedGain(self, bid):
        """
        计算预期收益
        """
        suite = self.player.posterior 
        total = 0
        for price, prob in sorted(suite.Items()):
            gain = self.Gain(bid, price)
            total += prob* gain
        return total
    
    def Gain(self, bid, price):
        """
        bid: 出价
        price: 实际价格
        """
        if bid > price:
            return 0

        diff = price - bid
        prob = self.ProbWin(diff)
        if diff <= 250: # 250以内收货两个展柜
            return 2 * price * prob
        else:
            return price * prob
        
    def ProbWin(self, diff):
        """
        给出两者的差距， 算出获胜的可能性
        这个计算过程是以对手的角度进行的:
             对手计算的正是: 我出高的可能性 和 我出价超出diff的概率是多少
        """
        prob = (self.opponnet.ProbOverbid() + 
                self.opponnet.ProbWoseThan(diff))
        return prob


class Player(object):
    # 代表获胜的玩家
    n = 101 
    price_xs = np.linspace(0, 75000, n)

    def __init__(self, price, bids, diffs):
        self.pdf_price = bys.EstimatedPdf(price)
        self.cdf_diff = bys.MakeCdfFromList(diffs)

        mean_ = 0; std = np.std(diffs)
        self.pdf_error = bys.GaussianPdf(mean_, std)
    
    def ErrorDensity(self, error):
        """
        该错误出现的概率密度
        """
        return self.pdf_error.Density(error)
    
    def PmfPrice(self):
        """
        价格的质量密度分布
        """
        return self.pdf_price.MakePmf(self.price_xs)
    
    def CdfDiff(self):
        """
        diff的 概率密度分布
        """
        return self.cdf_diff
    
    def ProbOverbid(self):
        return self.cdf_diff(-1)
    
    def ProbWoseThan(self, diff):
        return 1 - self.cdf_diff.Prob(diff)
    
    def MakeBeliefs(self, guess):
        """
        根据估价计算后验概率
        """
        pmf = self.PmfPrice()
        self.prior = Price(pmf, self, name = 'prior')
        self.posterior = self.prior.Copy(name = 'posterior')
        self.posterior.Update(guess)
    
    def OptimalBid(self, guess, opponnet):
        """
        计算使期望收益最大化的出价
        guess :猜测
        opponnet 对手
        """
        self.MakeBeliefs(guess) # 获得后验概率
        calc = GainCalulator(self, opponnet)
        bids, gains = calc.ExpectedGains()
        gain, bid = max(zip(gains, bids))
        return bid, gain


# 明天早上看 明天 2020-07-04
def MakePlayers():
    data = ReadData(fil_name)
    data += ReadData(fil_name1)

    cols = zip(*data)
    price1, price2, bid1, bid2, diff1, diff2 = cols

    plyer1 = Player(price1, bid1, diff1)
    plyer2 = Player(price2, bid2, diff2)
    return plyer1, plyer2


def MakePlots(player1, player2):
    bysplot.Clf()
    bysplot.PrePlot(num=2)
    pmf1 = player1.PmfPrice()
    pmf1.name = 'showcase 1'

    pmf2 = player1.PmfPrice()
    pmf2.name = 'showcase 2'
    bysplot.Pmfs([pmf1, pmf2])
    bysplot.Save(root = 'price1',
            xlabel = 'price ($)',
            ylabel = 'PDF',
            format = FROMATS)
    
    # plot the historical distribution of underness for both players
    bysplot.Clf()
    bysplot.PrePlot(num=2)
    cdf1 = player1.CdfDiff()
    cdf1.name = 'player 1'
    cdf2 = player2.CdfDiff()
    cdf2.name = 'player 2'

    print('Player median', cdf1.Percentile(50))
    print('Player median', cdf2.Percentile(50))

    print('Player 1 overbids', player1.ProbOverbid())
    print('Player 2 overbids', player2.ProbOverbid())

    bysplot.Cdfs([cdf1, cdf2])
    bysplot.Save(root='price2',
                xlabel='diff ($)',
                ylabel='CDF',
                formats=FORMATS)
    
def PlotExpectedGains(guess1 = 200000, guess2=400000):
    player1, player2 = MakePlayers()
    guesses = np.linspace(15000, 60000, 21)

    res = []
    for guess in guesses:
        player.MakeBeliefs(guess)

        mean_ = player1.posterior.Mean()
        mle = player1.posterior.MaximumLikelihood()

        calc = GainCalculator(player1, player2)
        bids, gains = calc.ExpectedGains()
        gain, bid = max(zip(gains, bids))

        res.append((guess, mean, mle, gain, bid))

    guesses, means, _mles, gains, bids = zip(*res)
    
    bysplot.PrePlot(num=3)
    pyplot.plot([15000, 60000], [15000, 60000], color='gray')
    bysplot.Plot(guesses, means, label='mean')
    #bysplot.Plot(guesses, mles, label='MLE')
    bysplot.Plot(guesses, bids, label='bid')
    bysplot.Plot(guesses, gains, label='gain')
    bysplot.Save(root='price6',
                   xlabel='guessed price ($)',
                   formats=FORMATS)


def TestCode(calc):
    """Check some intermediate results.
    calc: GainCalculator
    """
    # test ProbWin
    for diff in [0, 100, 1000, 10000, 20000]:
        print(diff, calc.ProbWin(diff))
    print

    # test Return
    price = 20000
    for bid in [17000, 18000, 19000, 19500, 19800, 20001]:
        print(bid, calc.Gain(bid, price))
    print


def main():
    PlotExpectedGains()
    PlotOptimalBid()



if __name__ == '__main__':
    main()



