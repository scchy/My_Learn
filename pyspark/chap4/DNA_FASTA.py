# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/scchy/pyspark-algorithms/blob/master/code/chap04/DNA-FASTA-V1/run_dna_base_count_ver_1.py
# create date: 2019-12-23
# function: DNA_FASTA_V1
# data: https://github.com/scchy/pyspark-algorithms/blob/master/code/chap04/data/sample.fasta

import sys, os
from pyspark.sql import SparkSession
from pyspark import StorageLevel
# 增加defaultdict
from collections import defaultdict

def process_FASTA_record(fasta_record):
    key_value_list = []

    if (fasta_record.startswith(">")):
        key_value_list.append(("z", 1))
    else:
        chars = fasta_record.lower()
        #print("chars=", chars)
        for c in chars:
            #print("c=", c)
            key_value_list.append((str(c), 1))
    return key_value_list


def process_FASTA_as_hashmap(fasta_record):
    if (fasta_record.startswith(">")):
        return [("z", 1)]
    #
    hashmap = defaultdict(int)
    chars = fasta_record.lower()
    #
    for c in chars:
        hashmap[c] += 1

    key_value_list = [(k, v) for k, v in hashmap.items()]

    print("key_value_list=", key_value_list)
    """
    key_value_list= [('a', 8), ('t', 3), ('c', 6)]
    key_value_list= [('a', 6), ('g', 3), ('c', 8), ('t', 4)]
    key_value_list= [('g', 18), ('c', 11), ('a', 13), ('t', 10)]
    key_value_list= [('g', 2), ('a', 4), ('t', 2), ('c', 4)]
    key_value_list= [('c', 8), ('g', 2), ('t', 7), ('a', 17)]
    key_value_list= [('a', 3), ('g', 10), ('c', 4), ('t', 7)]
    key_value_list= [('g', 16), ('c', 16), ('a', 18), ('t', 10), ('n', 2)]
    """
    return  key_value_list



if __name__ == '__main__':
    spark = SparkSession.builder.appName('DNA_FASTA').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sample.fasta'

    rdd = spark.sparkContext.textFile(fil_name)
    print("rdd.count = ",  rdd.count())
    print("rdd.collect() = ",  rdd.collect())
    pair_rdd = rdd.flatMap(lambda r: process_FASTA_record(r)) # piplinerdd

    pair_rdd1 = rdd.flatMap(lambda r: process_FASTA_as_hashmap(r))
    print(pair_rdd1.collect())
    freq_rdd = pair_rdd.groupByKey().mapValues(lambda x: sum(x))
    # freq_rdd = pair_rdd.reduceBykey(lambda x,y: x+y)
    print('freq_rdd.collect(): ', freq_rdd.collect())
    
    spark.stop()
