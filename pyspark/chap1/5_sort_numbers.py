# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap01/sort_numbers.py
# create date: 2019-12-13
# function: sort_numbers
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap01/sample_numbers.txt

import sys, os
from pyspark.sql import SparkSession


def debug_file(fil_path):
    f = open(fil_path, 'r')
    fil_con = f.read()
    print('file_contents = \n' + fil_con)
    f.close()



if __name__ == '__main__':
    spark = SparkSession.builder.appName('sort_numbers').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sample_numbers.txt'

    records = spark.sparkContext.textFile(fil_name)
    print("rdd =",  records)
    print("rdd.count = ",  records.count())
    print("rdd.collect() = ",  records.collect())
    print("展平>>增加一列>>排序")
    sorted_cnt = records.flatMap(lambda rec: rec.split(' '))\
        .map(lambda n: (int(n), 1)).sortByKey()
    print(sorted_cnt.collect())
    output = sorted_cnt.collect()
    print("sorted numbers:",output)
    # for (num, unitcount) in output:
    #     print(num)
    spark.stop()
