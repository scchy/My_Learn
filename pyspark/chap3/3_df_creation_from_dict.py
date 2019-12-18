# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap03/dataframe_creation_from_dictionary.py
# create date: 2019-12-18
# function: dataframe_creation_from_dictionary
# data: 

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import Row




if __name__ == '__main__':
    spark = SparkSession.builder.appName('dataframe_creation_dict').getOrCreate()

    mydict = {'A': '1', 'B': '2', 'D': '8', 'E': '99'}
    df = spark.createDataFrame(mydict.items(), ['key', 'value'])
    print("df.count = ",  df.count())
    print("df.collect() = ",  df.collect())
    df.show()
    df.printSchema()
    spark.stop()
