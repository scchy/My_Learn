# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap03/word_count.py
# create date: 2019-12-19
# function: word_count
# data: 

import sys, os
from pyspark import SparkContext, SparkConf

def wordcount(sc: SparkContext, input_path: str) -> str:
    rdd = sc.textFile(input_path)
    word_rdd = rdd.flatMap(lambda l: l.split(' '))
    pair_rdd = word_rdd.map(lambda word: (word, 1))
    print(pair_rdd.reduceByKey(lambda a, b: a + b).collect())



if __name__ == '__main__':
    spk_conf = SparkConf()
    spk_conf.setAppName('WordCount').set('spark.executor.memory', '500M')
    spk_conf.set('spark.cores.max', 4)
    try:
        sc = SparkContext(conf = spk_conf) 
        input_path = r'D:\My_Learn\pyspark\chap1\sample.txt'
    except:
        print ("Failed to connect!")
        print(sys.exc_info()[0])
    
    # Execute word count
    wordcount(sc, input_path) 
