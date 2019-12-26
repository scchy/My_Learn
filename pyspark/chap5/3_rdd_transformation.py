# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap05/rdd_transformation_cartesian.py
# create date: 2019-12-25
# function: rdd_transformation
# data: 

import sys, os
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from collections import defaultdict


def create_pair(t3):
    return (t3[0], int(t3[2]))


def create_pair_gp(t3):
    # t3 = (name, city, number)
    name = t3[0]
    city = t3[1]
    number = t3[2]
    return (name, (city, number))

def age_filter_26(t3: tuple) -> bool:
    age = t3[1]
    if (age > 26):
        return True
    else:
        return False

def tokenize(record):
    tokens = record.split()
    mylist = []
    for word in tokens:
        if len(word) > 2:
            mylist.append(word)
    return mylist


def minmax(iterator):
    first_time = 0
    for x in iterator:
        if (first_time == 0):
            min_ = x
            max_ = x
            first_time = 1
        else:
            if x > max_:
                max_ = x
            if x < min_:
                min_ = x
        #end-if
    #end-for
    return (min_, max_)


def min_max_count(iterator: iter) -> list:
    try:
        n = 0
        for ite_i in iterator:
            n += 1
            numbers = ite_i.split(",")
            # convert strings to integers
            numbers = list(map(int, numbers))
            print(numbers)
            if n == 1 :
                local_min = min(numbers)
                local_max = max(numbers)
                local_count = len(numbers)
            else: # 处理partition小于时
                if local_min > min(numbers):
                    local_min = min(numbers)
                if local_max < min(numbers):
                    local_max = min(numbers)
                local_count += len(numbers)
        return [(local_min, local_max, local_count)]
    except : # 处理partition 为空时
        # WHERE min > max to filter it out later
        return [(1, -1, 0)]



if __name__ == '__main__':
    spark = SparkSession.builder.appName('rdd_transformation').getOrCreate()
    # fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sp1.fastq'
    b = [('p', 50), ('x', 60), ('y', 70), ('z', 80) ]
    a = [('a', 2), ('b', 3), ('c', 4)]
    rdd_a = spark.sparkContext.parallelize(a)
    rdd_b = spark.sparkContext.parallelize(b)
    # -------------------------------
    # 1- cartesian()
    # -------------------------------
    cart = rdd_a.cartesian(rdd_b)
    print("cart.collect() = ", cart.collect())

    # -------------------------------
    # 2- combineByKey()
    # -------------------------------
    list_of_tuples= [('alex','Sunnyvale', 25), ('alex','Sunnyvale', 33), ('alex','Sunnyvale', 45)
    , ('alex','Sunnyvale', 63),('mary', 'Ames', 22), ('mary', 'Cupertino', 66), ('mary', 'Ames', 20), ('bob', 'Ames', 26)]
    rdd_cb = spark.sparkContext.parallelize(list_of_tuples)
    rdd_cb1 = rdd_cb.map(lambda x:create_pair(x))
    rdd_cbed = rdd_cb1.combineByKey(
        lambda value_: (value_, value_, 1),
        lambda comb, value_: (min(comb[0], value_), max(comb[1], value_), (comb[2] + 1)), # comb default (0,0,0)
        lambda cb_tup1, cb_tup2: (min(cb_tup1[0], cb_tup2[0]), max(cb_tup1[1], cb_tup2[1]), cb_tup1[2] + cb_tup2[2])
    )
    print("rdd_cbed.collect() = ", rdd_cbed.collect())

    # -------------------------------
    # 3- filter()
    # -------------------------------
    rdd_f = rdd_cb1.filter(lambda x: x[1] > 26)
    print("rdd_f.collect() = ", rdd_f.collect())
    rdd_f1 = rdd_cb1.filter(age_filter_26)
    print("func_bool_filter", rdd_f1.collect())

    # -------------------------------
    # 4- flatmap()
    # -------------------------------
    list_of_strings = ['of', 'a fox jumped',
                       'fox jumped of fence', 'a foxy fox jumped high']
    rdd_flat = spark.sparkContext.parallelize(list_of_strings)
    rdd_flated = rdd_flat.flatMap(lambda rec: tokenize(rec))
    print("rdd_flated.collect() = ", rdd_flated.collect())

    # -------------------------------
    # 5- groupByKey()
    # -------------------------------
    rdd_gp = rdd_cb.map(lambda x: create_pair_gp(x))\
                    .groupByKey()\
                    .mapValues(lambda values: list(values))

    print("rdd_gp.collect() = ", rdd_gp.collect())

    # -------------------------------
    # 6- join
    # -------------------------------
    source_pairs = [(1, "u"), (1, "v"), (2, "a"), (3, "b"), (4, "z1")]
    source = spark.sparkContext.parallelize(source_pairs)

    other_pairs = [(1, "x"), (1, "y"), (2, "c"), (2, "d"), (3, "m"), (8, "z2")]
    other = spark.sparkContext.parallelize(other_pairs)
    joined = source.join(other)
    # [(1, ('u', 'x')), (1, ('u', 'y')), (1, ('v', 'x')), (1, ('v', 'y')), (2, ('a', 'c')), (2, ('a', 'd')), (3, ('b', 'm'))]
    print("joined.collect(): ", joined.collect()) 

    # -------------------------------
    # 7- mapPartitions
    # -------------------------------
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    rdd = spark.sparkContext.parallelize(numbers, 3) # 3 partition
    print(rdd.collect())
    rdd.foreachPartition(lambda x: [print(i) for i in x])
    minmax_rdd = rdd.mapPartitions(minmax)
    print(minmax_rdd.collect())
    ## 7.12空的partition
    numbers = ["10,20,3,4",
               "3,5,6,30,7,8",
               "4,5,6,7,8",
               "3,9,10,11,12",
               "6,7,13",
               "5,6,7,12",
               "5,6,7,8,9,10",
               "11,12,13,14,15,16,17"]
    print("numbers = ", numbers)

    rdd_em = spark.sparkContext.parallelize(numbers, 10)
    # print("rdd_em.getNumPartitions() = ", rdd_em.getNumPartitions())
    # rdd_em.foreachPartition(lambda x:  print(x))
    # rdd_em.foreachPartition(lambda x: [print(i) for i in x])

    min_max_count_rdd = rdd_em.mapPartitions(min_max_count)
    print("min_max_count_rdd.collect() = ", min_max_count_rdd.collect())
    
    # -------------------------------
    # 8- sortBy
    # -------------------------------
    pairs = [(10, "z1"), (1, "z2"), (2, "z3"), (9, "z4"), (3, "z5"), (4, "z6"), (5, "z7"), (6, "z8"), (7, "z9")]
    rdd_st = spark.sparkContext.parallelize(pairs)
    print('值排序，倒序', rdd_st.sortBy(lambda x:x[1], ascending = False).collect())
    print('主键排序，倒序', rdd_st.sortByKey(ascending=False).collect())
    # -------------------------------
    # 9- takeOrdered
    # -------------------------------
    bottom3 = rdd_st.takeOrdered(3, key=lambda x: -x[0]) # return list
    print("bottom3 = ", bottom3)

    spark.stop()
    
