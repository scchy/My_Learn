# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/dataframe_creation_with_explicit_schema.py
# create date: 2020-01-02
# function: dataframe creation 2
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/emps_with_header.txt

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, exp, rand,  expr
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField

from pyspark.sql import Row
import pandas as pd

def m_sqrt(n):
    return n * n



if __name__ == '__main__':
    spark = SparkSession.builder.appName('df_with_explicit_schema').getOrCreate()

    # ---------------------------
    # 7- df create with explicit schema
    # ---------------------------
    emps_schema = StructType([
        StructField('id', IntegerType(), True),
        StructField('name', StringType(), True),
        StructField("salary", IntegerType(), True),
        StructField("dept", StringType(), True)
    ])
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\emps_no_header.txt'

    emps_df = spark.read.csv(fil_name, schema = emps_schema)
    emps_df.show()
    emps_df.printSchema()
    emps_df.createOrReplaceTempView('emp_table')
    # SQL
    df_show1 = spark.sql('select * from emp_table where id > 1002')
    print("df_show1=", df_show1.show())
    df_show2 = spark.sql('select dept, count(*) from emp_table group by dept')
    print("df_show2=", df_show2.show())

    # ---------------------------
    # 8- crosstab
    # ---------------------------
    ## 类似excel透视  但只显示成对的频率
    emps_df.crosstab('name', 'dept').show()
    data = [(1, 1), (1, 2), (2, 1), (2, 1), (2, 3), (3, 2), (3, 3), (4, 4)]
    columns = ["key", "value"]
    df_cr = spark.createDataFrame(data, columns)
    df_cr.crosstab('key', 'value').show()

    # ---------------------------
    # 9- 删除重复行
    # ---------------------------
    data = [(100, "a", 1.0),
            (100, "a", 1.0),
            (200, "b", 2.0),
            (300, "c", 3.0),
            (300, "c", 3.0),
            (400, "d", 4.0)]

    columns = ("id", "code", "scale")
    df_dp = spark.createDataFrame(data, columns)
    df_dp.dropDuplicates().show()
    spark.stop()

