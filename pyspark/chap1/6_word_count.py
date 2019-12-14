# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap01/word_count.py
# create date: 2019-12-14
# function: word_count
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap01/sample.txt

import sys, os
from pyspark.sql import SparkSession
from pyspark import SparkConf 
from pyspark import SparkContext


def word_count(sc, fil_name):
    rdd = sc.textFile(fil_name)
    print(rdd.collect())

    words_rdd = rdd.flatMap(lambda l: l.split(' '))
    print(words_rdd.collect())

    pair_rdd = words_rdd.map(lambda w: (w, 1))
    print(pair_rdd.collect())

    freq_rdd = pair_rdd.reduceByKey(lambda x, y: x + y)
    print(freq_rdd.collect())


if __name__ == '__main__':
    conf = SparkConf().setAppName('WordCount')
    conf.set('spark.executor.memory', '500M')
    conf.set('spark.cores.max', 4)
    try:
        sc = SparkContext(conf = conf)
        fil_name = r'D:\My_Learn\pyspark\chap1\sample.txt'
    except:
        print('Failed to connect!')
        print(sys.exc_info()[0])
    
    word_count(sc, fil_name)
