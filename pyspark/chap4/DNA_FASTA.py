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


