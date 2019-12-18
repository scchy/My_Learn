# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap03/dataframe_creation_from_collection.py
# create date: 2019-12-18
# function: dataframe_creation_from_collection
# data: 

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql import Row
from collections import OrderedDict



def covert_to_row(d: dict) -> Row:
    return Row(**OrderedDict(sorted(d.items())))

""" 申明式编写函数
def str_add(a: int, b: int) -> str:
    return str(a) + str(b)

str_add(10, 4) # '104'
"""

if __name__ == '__main__':
    spark = SparkSession.builder.appName('Wod-Count-App').getOrCreate()

    list_of_pairs = [("alex", 1), ("alex", 5), ("bob", 2),
                     ("bob", 40), ("jane", 60), ("mary", 700), ("adel", 800)]
    df1 = spark.createDataFrame(list_of_pairs)
    print("df1.collect() = ",  df1.collect())
    df1.show()
    df1.printSchema()

    # 增加特征名
    column_names = ["name", "value"]
    df2 = spark.createDataFrame(list_of_pairs, column_names)
    print("df1.collect() = ",  df2.collect())
    df2.show()
    df2.printSchema()

    # 建立结构再创表
    schema = StructType([
        StructField('name', StringType(), True),
        StructField('age', IntegerType(), True)
    ])
    df3 = spark.createDataFrame(list_of_pairs, schema)
    print("df1.collect() = ",  df3.collect())
    df3.show()
    df3.printSchema()
 
    # dict rows
    list_of_elements = [{"col1": "value11", "col2": "value12"}, {
        "col1": "value21", "col2": "value22"}, {"col1": "value31", "col2": "value32"}]
    df4 = spark.sparkContext.parallelize(list_of_elements).map(covert_to_row).toDF()
    print("df4 = ",  df4)
    print("df4.count = ",  df4.count())
    print("df4.collect() = ",  df4.collect())
    df4.show()
    df4.printSchema()

    spark.stop()
