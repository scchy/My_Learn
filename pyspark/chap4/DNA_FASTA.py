# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/scchy/pyspark-algorithms/blob/master/code/chap04/DNA-FASTA-V1/run_dna_base_count_ver_1.py
# create date: 2019-12-23
# function: DNA_FASTA_V1
# data: https://github.com/scchy/pyspark-algorithms/blob/master/code/chap04/data/sample.fasta

import sys, os
from pyspark.sql import SparkSession
from pyspark import StorageLevel


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


if __name__ == '__main__':
    spark = SparkSession.builder.appName('DNA_FASTA').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sample.fasta'

    rdd = spark.sparkContext.textFile(fil_name)
    print("rdd.count = ",  rdd.count())
    print("rdd.collect() = ",  rdd.collect())
    pair_rdd = rdd.flatMap(lambda r: process_FASTA_record(r)) # piplinerdd
    freq_rdd = pair_rdd.groupByKey().mapValues(lambda x: sum(x))
    # freq_rdd = pair_rdd.reduceBykey(lambda x,y: x+y)
    print('freq_rdd.collect(): ', freq_rdd.collect())
    
    spark.stop()
