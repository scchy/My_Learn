# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/partition_data_by_customer_and_year.py
# create date: 2020-01-06
# function: partition_data_by_customer_and_year
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/customers.txt

import sys, os
from pyspark.sql import SparkSession


if __name__ == '__main__':
    spark = SparkSession.builder.appName('df_with_statistical_data').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\customers.txt'

    df = spark.read.option('delimiter', ',').option('inferSchema', 'true').csv(fil_name).toDF(
        'customer_id', 'year', 'transaction_id', 'transaction_value')
    df.show(truncate=False)
    # 分区输出
    # partition data by 'customer_id', and then by 'year'
    part_path = r'E:\Work_My_Asset\pyspark_algorithms\chap1\part' # 需要在lunix 下 文件命名 windows 不允许
    df.repartition('customer_id', 'year')\
        .write.partitionBy('customer_id', 'year')\
        .parquet(part_path)

    df2 = spark.read.parquet(part_path)
    df2.show(truncate=False)
    spark.stop()

