# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap08/datasource_json_reader_multi_line.py
# create date: 2020-02-02
# function: datasource_json_reader_multi_line
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap08/sample_multi_line.json


import sys, os
from pyspark.sql import SparkSession, SQLContext



if __name__ == '__main__':
    spark = SparkSession.builder.appName('datasource_json_reader_multi_line')\
        .getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sample_multi_line.json'
    # -----------------
    # 1- 读取多行形式存储的 json
    # -----------------
    df = spark.read.option('multiline', 'true').json(fil_name)
    print("df.collect() = ", df.collect())
    df.show(10, truncate=False)

    # -----------------
    # 2- 读取单行形式存储的 json
    # -----------------
    fil_name_single = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sample_single_line.json'
    df_single = spark.read.json(fil_name_single)
    print("df_single.collect() = ", df_single.collect())
    df_single.show(10, truncate=False)
    spark.stop()

