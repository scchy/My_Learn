# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap08/datasource_gzip_reader.py
# create date: 2020-01-15
# function: datasource_gzip_reader
# data: https://github.com/mahmoudparsian/pyspark-algorithms/edit/master/code/chap08/sample_with_header.csv

import sys, os
from pyspark.sql import SparkSession
from os.path import isfile, join




if __name__ == '__main__':
    spark = SparkSession.builder.appName(
        'datasource_gzip_reader').getOrCreate()
    gz_input_path = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sample_with_header.csv'
    # 不读取表头
    gzip_rdd = spark.sparkContext.textFile(gz_input_path).filter(lambda x: 'name' not in x)
    print("gzip_rdd.collect() : \n", gzip_rdd.collect())

    spark.stop()

