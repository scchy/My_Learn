# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/scchy/pyspark-algorithms/blob/master/code/chap04/DNA-FASTQ/dna_base_count_fastq.py
# create date: 2019-12-23
# function: dna_base_count_fastq
# data: https://github.com/scchy/pyspark-algorithms/blob/master/code/chap04/data/sp1.fastq

import sys, os
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from collections import defaultdict


def drop_3_records(rec):
    fst_char = rec[0]
    if fst_char in ['@', '+']:
        return False 
    non_DNA_letetrs = set('-+*/<>=.:@?;0123456789bde')
    if any((c in non_DNA_letetrs) for c in rec):
        # print('rec=', rec)
        return False
    else:
        return True


def process_FASTQ_partition(iter_in):
    hashmap = defaultdict(int)
    for line in iter_in:
        hashmap['z'] += 1
        chars = line.lower()
        for c in chars:
            hashmap[c] += 1
    key_value_list = [(k, v) for k, v in hashmap.items()]

    return  key_value_list




if __name__ == '__main__':
    spark = SparkSession.builder.appName('DNA_FASTA').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sp1.fastq'

    rdd = spark.sparkContext.textFile(fil_name).map(lambda rec: rec.lower())
    print("rdd.count = ",  rdd.count())
    print("rdd.collect() = ",  rdd.collect())
    dna_seqs = rdd.filter(drop_3_records) # 仅拿DNA的列
    pairs_rdd = dna_seqs.mapPartitions(process_FASTQ_partition)
    print(pairs_rdd.collect())
    freq_rdd = pairs_rdd.groupByKey().mapValues(lambda x: sum(x))
    print('freq_rdd.collect(): ', freq_rdd.collect())

    spark.stop()
