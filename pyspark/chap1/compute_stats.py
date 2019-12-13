# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap01/compute_stats.py
# create date: 2019-12-13
# function: 
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap01/url_frequencies.txt

import sys, os
from pyspark.sql import SparkSession
from operator import add
import  statistics as stat

def compute_stats(num_dt):
    avg = stat.mean(num_dt)
    median = stat.median(num_dt)
    std = stat.stdev(num_dt)
    return avg, median, std

def create_pair(record):
    tokens = record.split(',')  # <2>
    url_address = tokens[0]
    frequency = int(tokens[1])
    return (url_address, frequency)  # <3>

if __name__ == '__main__':
    spark = SparkSession.builder.appName('compute_stats').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\url_frequencies.txt'

    print("读取[rdd]>>筛选>>(map)拆分[rdd]>>groupby计算(mapVaues)")
    results = spark.sparkContext.textFile(fil_name)\
        .filter(lambda record: len(record) > 5)\
        .map(create_pair)\
        .groupByKey()\
        .mapValues(compute_stats)

    results.collect()
    spark.stop()
