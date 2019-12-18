# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap03/dataframe_creation_from_rdd.py
# create date: 2019-12-19
# function: dataframe_creation_from_rdd
# data: 

import sys, os
from pyspark.sql import SparkSession
from pyspark import Row



if __name__ == '__main__':
    spark = SparkSession.builder.appName('dataframe_creation_from_rdd').getOrCreate()
    list_of_tuples = [('alex','Sunnyvale', 25), ('mary', 'Cupertino', 22), ('jane', 'Ames', 20), ('bob', 'Stanford', 26)]
    rdd = spark.sparkContext.parallelize(list_of_tuples)
    print("rdd = ", rdd)
    print("rdd.count() = ", rdd.count())
    print("rdd.collect() = ", rdd.collect())

    people = rdd.map(lambda x: Row(name=x[0], city=x[1], age=int(x[2])))
    print('people = ', people)
    print('people.count() = ', people.count())
    print('people.collect() = ', people.collect())

    df = spark.createDataFrame(people)
    print("df = ", df)
    print("df.count() = ", df.count())
    print("df.collect() = ", df.collect())
    df.show()
    df.printSchema()

    spark.stop()
