# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap01/rdd_creation_from_csv.py
# create date: 2019-12-13
# function: rdd_creation_from_csv
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap01/name_city_age.csv

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType
import pyspark.sql.functions as sql_func


def debug_file(fil_path):
    f = open(fil_path, 'r')
    fil_con = f.read()
    print('file_contents = \n' + fil_con)
    f.close()


def create_pair(record):
    tokens = record.split(",")
    city = tokens[1]
    age = int(tokens[2])
    return (city, (age, 1))

def add_pair(a, b):
    return a[0]+b[0], a[1]+b[1]

if __name__ == '__main__':
    spark = SparkSession.builder.appName('rdd_creation_from_csv').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\name_city_age.csv'

    rdd = spark.sparkContext.textFile(fil_name)
    print("rdd =",  rdd)
    print("rdd.count = ",  rdd.count())
    print("rdd.collect() = ",  rdd.collect())
    average_per_city  = rdd.map(create_pair).reduceByKey(lambda x, y: add_pair(x, y))\
        .mapValues(lambda pair : float(pair[0]) / float(pair[1]))
    print(average_per_city.collect())
    spark.stop()
