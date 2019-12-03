# python 3.6
# Author:              Scc_hy
# Create date:         2019-12-3
# Function:            pyspark dataframe example



import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *

# 数据在  https://github.com/mahmoudparsian/pyspark-algorithms/tree/master/code/chap01 下载
base_root = r'E:\Work_My_Asset\pyspark_algorithms\chap1'
os.chdir(base_root)
fil_name = 'sample_people.json'

def debg_file(input_path):
    f = open(input_path, 'r')
    fil_contents = f.read()
    print("fil_contents = \n" + fil_contents)
    f.close()


debg_file(fil_name)
