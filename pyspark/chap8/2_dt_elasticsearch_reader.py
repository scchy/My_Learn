# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap08/datasource_elasticsearch_reader.py
## 写入看： datasource_elasticsearch_writer.py
# create date: 2020-01-15
# function: datasource_csv_write
# data: https://github.com/mahmoudparsian/pyspark-algorithms/edit/master/code/chap08/sample_with_header.csv

import sys, os
from pyspark.sql import SparkSession
from os.path import isfile, join




if __name__ == '__main__':
    spark = SparkSession.builder.appName(
        'datasource_elasticsearch_reader').getOrCreate()
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sample_with_header.csv'

    es_read_conf = {
        # specify the node that we are sending data to
        'es.nodes': fil_name,
        'es.port'  : '9200',
        'es.resource': 'testindex/testdoc'
    }
    es_rdd = spark.sparkContext.newAPIHadoopRDD(
        inputFormatClass = 'org.elasticsearch.hadoop.mr.EsInputFormat',
        keyClass = 'org.apache.hadoop.io.NullWritable',
        valueClass = 'org.elasticsearch.hadoop.mr.LinkedMapWriteable',
        conf = es_read_conf
    )
    print("es_rdd.collect() \n: ", es_rdd.collect())

    spark.stop()

