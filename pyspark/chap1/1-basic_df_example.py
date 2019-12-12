# python 3.6
# Author:              Scc_hy
# Create date:         2019-12-12
# Function:            pyspark dataframe example

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *


base_root = r'E:\Work_My_Asset\pyspark_algorithms\chap1'#'/home/app_social/spk_d_scc'
os.chdir(base_root)
fil_name = 'sample_people.json'

def debg_file(input_path):
    f = open(input_path, 'r')
    fil_contents = f.read()
    print("fil_contents = \n" + fil_contents)
    f.close()


# debg_file(fil_name)
def create_pair(record):
    token = record.split(',')
    city = token[1]
    age = int(token[2])
    return (city, (age, 1))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: basic_dataframe_example.py <json-file>", file=sys.stderr)
        exit(-1)
    spark = SparkSession.builder.appName("basic_dataframe_example").getOrCreate()
    print("spark=",  spark)

    # read name of input file
    json_input_path = sys.argv[1] # os.path.join(base_root,fil_name)
    print("JSON input path : ", json_input_path)
    debg_file(json_input_path)
    df = spark.read.json(json_input_path)
    print("df.show():")
    df.show()

    df.printSchema()
    df.select('name').show()
    df.select(df['name'], df['age'] + 1).show()
    # where
    df.filter(df['age'] > 23).show()
    # group
    df.groupBy('age').count().show()
    # register the DataFrame as a SQL TEMPORARY VIEW
    df.createOrReplaceTempView('people') # df 命名成people
    sql_df = spark.sql('select * from people')
    sql_df.show()

    # Register the df as a globeltemporary view
    df.createGlobalTempView('people')
    spark.sql('select * from global_temp.people').show()

    spark.stop()
