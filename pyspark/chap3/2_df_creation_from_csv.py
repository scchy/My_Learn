# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap03/dataframe_creation_from_csv_no_header.py
# create date: 2019-12-18
# function: dataframe_creation_csv_no_header
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap03/kv_no_header.txt

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import Row




if __name__ == '__main__':
    spark = SparkSession.builder.appName('dataframe_creation_csv_no_header').getOrCreate()
    filname = r'E:\Work_My_Asset\pyspark_algorithms\chap1\kv_no_header.txt'
    list_of_pairs = [("alex", 1), ("alex", 5), ("bob", 2),
                     ("bob", 40), ("jane", 60), ("mary", 700), ("adel", 800)]
    # 暂且不设置数据结构
    ##  .option("header","true") 加载头 kv_with_header.txt
    df = spark.read.format('csv').option('header', 'false')\
        .option('inferSchema', 'true').load(filname) 
    print("df = ", df.collect())
    df.show()
    df.printSchema()
    df2 = df.selectExpr("_c0 as name", "_c1 as value")
    df2.show()
    spark.stop()
