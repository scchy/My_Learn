# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap01/dataframe_creation_from_csv.py
# create date: 2019-12-13
# function: dataframe_creation_from_csv
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

if __name__ == '__main__':
    spark = SparkSession.builder.appName('datafrme_creation_from_csv').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\name_city_age.csv'

    schema = StructType([
        StructField('name', StringType()),
        StructField('city', StringType()),
        StructField('age', DoubleType())
    ])

    df = spark.read.schema(schema).format('csv')\
        .option('header', 'false')\
        .option('inferSchema', 'true')\
        .load(fil_name)
    print("df.count() = ", df.count())
    print("df.collect() = ", df.collect())
    df.show()

    average_method1 = df.groupBy('city').agg(sql_func.avg('age').alias('average'))
    average_method1.show()

    df.createOrReplaceTempView('df_tbl')
    average_method2 = spark.sql("select city, avg(age) avg_age from df_tbl group by city")
    average_method2.show()
    spark.stop()
