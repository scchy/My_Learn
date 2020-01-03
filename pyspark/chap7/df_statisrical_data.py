# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/dataframe_with_statistical_data.py
# create date: 2020-01-03
# function: dataframe_with_statistical_data
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/life_expentancy.txt

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, IntegerType, DoubleType
from pyspark.sql.types import StructType, StructField


if __name__ == '__main__':
    spark = SparkSession.builder.appName('df_with_statistical_data').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\life_expentancy.txt'
    custom_schema = StructType([
        StructField('country', StringType(), True),
        StructField('life_exp', DoubleType(), True),
        StructField('region', StringType(), True)
    ])
    df = spark.read.option('delimiter', ',').csv(fil_name, schema = custom_schema)
    df.show(truncate=False)
    # ---------------------------
    # 1- summary
    # ---------------------------
    df.summary().select('life_exp').show()

    # ---------------------------
    # 2- approxQuantile
    ## 近似百分位 快速求解
    # ---------------------------
    quantileProbs = [0.01, 0.25, 0.5, 0.75, 0.99]
    relError = 0.01
    explore_dt = df.approxQuantile("life_exp", quantileProbs, relError)
    print("explore_dt : ", explore_dt)
    spark.stop()

