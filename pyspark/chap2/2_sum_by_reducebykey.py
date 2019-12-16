# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap02/sum_by_reducebykey.py
# create date: 2019-12-16
# function: sum_by_reducebykey
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap02/sum_by_groupbykey.log 

import sys, os
from pyspark.sql import SparkSession


def create_pair(record):
    tokens = record.split(',')
    return (str(tokens[0]), int(tokens[1]))


if __name__ == '__main__':
    spark = SparkSession.builder.appName('test:groupBykey()').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\kv.txt'

    records = spark.sparkContext.textFile(fil_name)
    print("rdd.getNumPartitions() = ", records.getNumPartitions())
    # 比 map().groupByKy().mapValues(lambda a :sum(a))要少个链
    res = records.map(create_pair).reduceByKey(lambda a, b: a+b)
    print("sum numbers:", res.collect())

    spark.stop()
