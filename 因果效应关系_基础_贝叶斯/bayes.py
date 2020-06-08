# -*- coding:utf-8 -*-
# python 3.6
# author: Scc_hy 
# create date: 2020-04-18
# Function： 贝叶斯常用的类
import copy
import math
import random 
from datetime import datetime
import numpy as np
import scipy

def get_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class _DictWrapper():
    """
    字典
    """
    def __init__(self, values = None, name = ''):
        """
        初始化分布
        """
        self.name = name
        self.d = {}
        self.log = False
        if values is None:
            return
        
        init_methods = [
            self.InitPmf,
            self.InitMapping,
            self.InitSequence,
            self.InitFailure,
        ]

        for method in init_methods:
            try:
                method(values)
                break
            except AttributeError:
                continue
        
        if len(self) > 0:
            self.Normalize()
    
    def Set(self, x, y=0):
        """ 
        给d字典赋值
        param x: 事件 object
        param y: 频率/频数 float/int
        """
        self.d[x] = y
    
    def InitSequence(self, values):
        """
        用相同值的序列初始化。
        param values : list 
        """
        for v in values:
            self.Set(v, 1)
    
    def InitMapping(self, values):
        """
        输入的是字典的时候
        param values: dict 
        """
        for  k, v in values.items():
            self.Set(k, v)
    
    def InitPmf(self, values):
        """
        输入的是 Pmf的时候 （概率质量函数）
        """
        for k, v in values.Items():
            self.Set(k, v)
        
    def InitFailure(self, values):
        raise ValueError('None of the initialization methods worked.')
    
    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def Copy(self, name=None):
        """
        复制一份
        """
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.name = name if name is not None else self.name
        return new

    def Scale(self, factor):
        """
        将值乘以一个系数
        """
        new = self.Copy()
        new.d.clear()
        for val, prob in self.Items():
            new.Set(val * factor, prob)
        return new
    
    def Log(self, m=None):
        """
        对数变换
        移除概率为0的值
        进行标准化，使得 log(prob) 最大为0
        """
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True    

        if m is None:
            m = self.MaxLike()
        
        for x, p in self.d.items():
            if p:
                self.Set(x, math.log(p /m))
            else:
                self.Remove(x)
        
    def Exp(self, m=None):
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            self.Set(x, math.exp(p - m))      

    def SetDict(self, d):
        """
        直接给定一个字典        
        """
        self.d = d
    
    def Values(self):
        """
        获取值 d.keys
        """
        return self.d.keys()

    def Items(self):
        """
        获取 key - value对
        """
        return self.d.items()

    def Render(self):
        """
        list(zip(*sorted(d.items())))
        [('a', 'b', 'c', 'd', 'e', 'f'), (0.1, 0.2, 0.9, 0.1, 0.35, 0.25)]
        """
        return zip(*sorted(self.Items()))

    def Print(self):
        for val, prob in sorted(self.d.items()):
            print(val, prob)
    
    def Incr(self, x, term=1):
        """
        增加某个事件的值 加法
        """
        self.d[x] = self.d.get(x, 0) + term
    
    def Mult(self, x, fator):
        """
        增加某个事件的值 乘法
        """
        self.d[x] = self.d.get(x, 0) * fator
    
    def Remove(self, x):
        """
        删除某事件
        """
        del self.d[x]
    
    def Total(self):
        """
        汇总事件的概率
        """
        return sum(self.d.values())
    
    def MaxLike(self):
        """
        取概率最大的值
        """
        return max(self.d.values())

    def GetDict(self):
        """Gets the dictionary."""
        return self.d


class Pmf(_DictWrapper):
    """
    概率质量函数
    值可以是任何散列的类型；概率为浮点型
    Pmfs不一定要标准化
    """
    def Prob(self, x, default=0):
        """
        获取事件的概率
        """
        return self.d.get(x, default)
    
    def Probs(self, xs:list) -> list:
        """
        获取一群事件的概率
        params xs:list
        """
        return [self.Prob(x) for x in xs]
    
    def MakeCdf(self, name=None):
        return MakeCdfFromPmf(self, name = name)
    
    def ProbGreater(self, x):
        """
        大于阈值x的所有概率项之和
        """
        return sum([v for k, v in self.d.items() if v > x])

    def ProbLess(self, x):
        """
        小于阈值x的所有概率项之和
        """
        return sum([v for k, v in self.d.items() if v < x])


    def __lt__(slef, obj):
        """
        less than 
        """
        if isinstance(obj, _DictWrapper):
            return PmfProbLess(self, obj)
        else:
            return self.ProbLess(obj)


    def __ge__(self, obj):
        """
        Greater than or equal.
        obj: number or _DictWrapper
        returns: float probability
        """
        return 1 - (self < obj)

    def __le__(self, obj):
        """Less than or equal.
        obj: number or _DictWrapper
        returns: float probability
        """
        return 1 - (self > obj)

    def __eq__(self, obj):
        """
        等于
        """
        if isinstance(obj, _DictWrapper):
            return PmfProbEqual(self, obj)
        else:
            return self.Prob(obj)

    def __ne__(self, obj):
        """
        不等于
        """
        return 1 - (self == obj)

    def Normalize(self, fraction=1.0):
        """
        标准化，使得概率总和为 fraction
        """
        if self.log:
            raise ValueError("Pmf is under a log transform")

        total = self.Total()
        if total == 0.0:
            raise ValueError('total probability is zero.')
            logging.warning('Normalize: total probability is zero.')
            return total
        
        f = fraction / total
        for x in self.d:
            self.d[x] *= f
        
        return total
    
    def Random(self):
        """
        随机选取一个元素
        """
        if len(self.d) == 0:
            raise ValueError('Pmf contains no values.')

        return random.choice(list(self.d.keys()))
    
    def Mean(self):
        """
        计算均值
        """
        mu = 0.0
        for x, p in self.d.items():
            mu += p * x
        return mu

    def Var(self, mu=None):
        """
        计算方差
        """
        if mu is None:
            mu=self.Mean()
        
        var= 0.0
        for k, v in self.d.items():
            var += p * (x - mu) ** 2
        return var
    
    def MaximumLikelihood(self):
        """
        Returns the value with the highest probability.
        """
        prob, val = max((prob, val) for val, prob in self.Items())
        return val

    def CredibleInterval(self, percentage=90):
        """
        计算置信区间
        param percentage: 百分比 0- 100
        """
        cdf = self.MakeCdf()
        return cdf.CredibleInterval(percentage)
    
    def AddPmf(self, otherPmf):
        """
        计算自己和另一个Pmf的和, 并返回新的类
        """
        out_pmf = Pmf()
        for k1, v1 in self.Items():
            for k_o, v_o in otherPmf.Items():
                out_pmf.Incr(k1+k_o, v1*v_o)
        return out_pmf
    
    def AddConstant(self, other):
        """
        对key增加 一个值, 并返回新的类
        """
        out_pmf = Pmf()
        for v1, p1 in self.Items():
            out_pmf.Set(v1 + other, p1)
        return out_pmf    

    def __add__(self, other):
        """
        功能同 AddPmf + AddConstant
        """
        try:
            return self.AddPmf(other)
        except AttributeError:
            return self.AddConstant(other)        

    def __sub__(self, other):
        """
        减去keys的值
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 - v2, p1 * p2)
        return pmf
    
    def Max(self, k):
        """
        Computes the CDF of the maximum of k selections from this dist
        """
        cdf = self.MakeCdf()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf
    

def MakeCdfFromPmf(pmf, name=None):
    if name == None:
        name = pmf.name
    return MakeCdfFromItems(pmf.Items(), name)

def MakeCdfFromItems(pmf_items, name):
    """
    make cdf from an unsorted sequence of (value, frequence) pairs
    """
    cnt = 0
    xs, cs =[], []
    for k, v in sorted(pmf_items):
        cnt += v
        xs.append(k)
        cs.append(cnt)
    
    ps = [c / cnt for c in cs] # 累计百分比
    cdf = Cdf(xs, ps, name)
    return cdf

def  MakePmfFromCdf(cdf, name=None):
    """
    累计分布到质量分布
    """
    if name is None:
        name = cdf.name
    
    pmf = Pmf(name=name)
    p_tmp = 0
    for xs, ps in cdf.Items():
        pmf.Incr(xs, ps - p_tmp)
        p_tmp = ps
    return pmf

class Cdf(object):
    """
    累积分布函数
    """

    def __init__(self, xs=None, ps=None, name=''):
        self.xs = [] if xs is None else xs
        self.ps = [] if ps is None else ps
        self.name = name
    
    def Copy(self, name=None):
        if name is None:
            name = self.name
        return Cdf(list(self.xs), list(self.ps), name)
       
    def MakePmf(self, name=None):
        """
        转换成 质量分布函数        
        """
        return MakePmfFromCdf(self, name=name)

    def Values(self):
        return self.xs

    def Items(self):
        return zip(self.xs, self.ps)
    
    def Append(self, x, p):
        """
        增加
        """
        self.xs.append(x)
        self.ps.append(p) 

    def Shift(self, term):
        """
        对xs偏移
        """
        new = self.Copy()
        new.xs = [x + term for x in self.xs]
        return new

    def Scale(self, factor):
        """
        xs 每个项乘以系数
        """
        new = self.Copy()
        new.xs = [x * factor for x in self.xs]
        return new
    
    def Prob(self, x):
        """
        返回对应概率
        """
        if x < self.xs[0]: 
            return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p   
    
    def Values(self, p):
        if p < 0 or p > 1:
            raise ValueError('Probability p must be in range [0, 1]')
        if p == 0: return self.xs[0]
        if p == 1: return self.xs[-1]
        index = bisect.bisect(self.ps, p)  

    def Percentile(self, p):
        """
        返回对应于百分位数p的值
        """
        return self.Values(p / 100)
    
    def Random(self):
        return self.Values(random.random())
    
    def Sample(self, n):
        return [self.Random() for i in range(n)]
    
    def Mean(self):
        """
        计算分布的期望
        """
        old_p, total = 0, 0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p*x
            old_P = new_p
        return total

    def Max(self, k):
        """
        选择K个值， 的分布
        """
        cdf = self.Copy()
        cdf.ps = [p**k for p in cdf.ps]
        return cdf 



# CDF还未结束
# 
# 
#
class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class Suite(Pmf):
    def Update(self, data):
        for p in self.Values():
            like = self.Likelihood(data, p)
            self.Mult(p, like)
        return self.Normalize()

    def Likelihood(self, data, p):
        raise UnimplementedMethodException()

    def Print(self):
        """Prints the hypotheses and their probabilities."""
        for hypo, prob in sorted(self.Items()):
            print(hypo, prob)


def MakePmfFromDict(d, name=''):
    pmf = Pmf(d, name)
    pmf.Normalize()
    return pmf


# Beta分布
class Beta(object):
    """
    Beta 分布
    """
    def __init__(self, alpha = 1, beta = 1, name = ''):
        """
        初始化分布
        """
        self.alpha = alpha
        self.beta = beta
        self.name = name 
    
    def Update(self, data):
        """
        Updata a Beta distribution
        param data: (heads:int, tail:int)
        """
        heads, tails = data
        self.alpha += heads
        self.beta += tails
    
    def Mean(self):
        return float(self.alpha) / (self.alpha + self.beta)

    def Random(self):
        return random.betavariate(self.alpha, self.beta)


    def Sample(self, n):
        return np.random.beta(self.alpha, self.beta, size=n)

    def EvalPdf(self, steps=101, name=''):
        """
        返回： 概率质量函数
        """
        if self.alpha < 1 or self.beta < 1:
            cdf = self.MakeCdf()
            pmf = self.MakePmf()
            return pmf
        
        xs = [i / (steps - 1) for i in range(steps)]
        probs = [self.EvalPdf(x) for x in xs] 
        pmf = MakePmfFromDict(dict(zip(xs, probs)), name)
        return pmf


    def MakcCdf(self, steps=101):
        """
        返回累计分布
        """
        xs = [i/(steps - 1) for i in range(steps)]
        ps = [scipy.special.betainc(self.alpha, self.beta, x) for x in xs]
        cdf = Cdf(xs, ps)
        return cdf



class Hist(_DictWrapper):
    """
    返回直方图，它是从值到频率的映射
    """
    def Freq(self, x):
        return self.d.get(x, 0)
    
    def Freqs(self, xs):
        return [self.Freq(x) for x in xs]

    def IsSubset(self, other):
        for val, freq in self.d.items():
            if freq > other.Freq(val):
                return False
        return True

    def Subtract(self, other):
        for val, freq in other.d.items():
            self.Incr(val, -freq) # 增加



def MakeHistFromList(t, name = ''):
    """
    绘制直方图
    """
    hist = Hist(name=name)
    [hist.Incr(x) for x in t]
    return hist




def MakePmfFromList(t, name = ''):
    """
    生成 质量概率分布
    t : sequence of number
    name: 命名pmf
    """
    hist = MakeHistFromList(t)
    d = hist.GetDict()
    pmf = Pmf(d, name)
    pmf.Normalize()
    return pmf


def SampleSum(dists, n):
    """
    dists: list of Pmf / Cdf
    n: 样本次数
    return: new Pmf of sums
    """
    pmf = MakePmfFromList(RandomSum(dists) for i in range(n))
    return pmf

def RandomSum(dists):
    """
    d_ 为pmf对象
    """
    # random.choice(list(self.d.keys()))
    total = sum(d_.Random() for d_ in dists)
    return total


# 最大值
def RandomMax(dists):
    total = max(dist.Random() for dist in dists)
    return total 

def SampleMax(dists, n ):
    pmf = MakePmfFromList(RandomMax(dists) for i in range(n) )
    return pmf

def PmfMax(pmf1, pmf2):
    res = Pmf()
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            res.Incr(max(v1, v2), p1*p2)
    return res 


def MakeMixture(pmf_lst, weight_lst,name='mix'):
    """
    由 pmf_lst 与 weight_lst 组成混合分布   
    param pmf_lst: pmf类的 列表  
    param weight_lst: pmf类的权重  
    """
    mix = Pmf(name=name)
    for pmf_tmp, tz_nm in zip(pmf_lst, weight_lst):
        for o, prob in pmf_tmp.Items():
            mix.Incr(o, tz_nm * prob)
    mix.Normalize()
    return mix
