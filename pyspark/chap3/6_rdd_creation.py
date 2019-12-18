# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap03/rdd_creation_from_*.py
# create date: 2019-12-19
# function: rdd creation
# data: 

import sys, os
from pyspark.sql import SparkSession
from pyspark import Row



if __name__ == '__main__':
    spark = SparkSession.builder.appName('dataframe_creation_from_rdd').getOrCreate()

    #=====================================
    # 1- RDD from collection
    #=====================================
    ## list value
    list_of_strs = ['alx', 'bob', 'jane', 'mary', 'adel']
    rdd1 = spark.sparkContext.parallelize(list_of_strs)
    ## list pairs
    list_of_pair = [("alex", 1), ("alex", 3), ("alex", 9), ("alex", 10), ("bob", 4), ("bob", 8)]
    rdd2 = spark.sparkContext.parallelize(list_of_pair)
    ### reduce
    rdd2_add = rdd2.reduceByKey(lambda x, y: x+y)
    print(rdd2_add.collect())
    ### groupby 
    print(rdd2.groupByKey().mapValues(lambda v: list(v)).collect())

    #=====================================
    # 2- RDD from dict 
    #=====================================
    d = {"key1":"value1","key2":"value2","key3":"value3"} 
    rdd_f_dict = spark.sparkContext.parallelize(d.items())
    print("rdd_f_dict.collect() = ",  rdd_f_dict.collect())
    print("rdd_f_dict.count = ",  rdd_f_dict.count())  

    #=====================================
    # 3- RDD from df
    #=====================================
    list_of_pairs = [("alex", 1), ("alex", 5), ("bob", 2), ("bob", 40), ("jane", 60), ("mary", 700), ("adel", 800) ]
    df = spark.createDataFrame(list_of_pairs)
    df.show()
    # Convert an existing DataFrame to an RDD
    rdd = df.rdd
    print("rdd.collect() = ",  rdd.collect())

    #=====================================
    # 4- RDD from dir
    #=====================================
    dir_name = r'D:\My_Learn\pyspark\chap1\sample_dir'
    print(os.listdir(dir_name))

    rdd_dir = spark.sparkContext.textFile(dir_name+'/*.txt')
    print("rdd_dir.collect() = ",  rdd_dir.collect())
    ## filter
    filtered = rdd_dir.filter(lambda a: a.find('3') != -1) # which contain 3
    print("filtered.collect() = ",  filtered.collect()) 
    spark.stop()
