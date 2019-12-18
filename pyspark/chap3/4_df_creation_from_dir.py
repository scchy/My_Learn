# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap03/dataframe_creation_from_dictionary.py
# create date: 2019-12-18
# function: dataframe_creation_from_directory
# data: https://github.com/mahmoudparsian/pyspark-algorithms/tree/master/code/chap03/sample_dir2

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import Row




if __name__ == '__main__':
    spark = SparkSession.builder.appName('dataframe_creation_dict').getOrCreate()
    dir_root = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sample_dir2'
    print(os.listdir(dir_root))
    # 用于相同结构分块存储的数据
    df = spark.read.format('csv').option('header','false')\
        .option('inferSchema', 'true').load(dir_root+'/*.csv')
    print("df.count = ",  df.count())
    df.show()
    df.printSchema()
    spark.stop()
