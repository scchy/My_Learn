# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap05/dataframe_action_describe.py
# create date: 2019-12-24
# function: dataframe_action
# data: 

import sys, os
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from collections import defaultdict


def create_pair(t3):
    return (t3[0], int(t3[2]))

def create_pair_sumcnt(t3):
    name = t3[0]
    number = int(t3[2])
    return (name, (number, 1))


if __name__ == '__main__':
    spark = SparkSession.builder.appName('df_action').getOrCreate()
    # fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\sp1.fastq'
    pairs = [(10,"z1"), (1,"z2"), (2,"z3"), (9,"z4"), (3,"z5"), (4,"z6"), (5,"z7"), (6,"z8"), (7,"z9")]
    df = spark.createDataFrame(pairs, ['number', 'name'])
    df.show()
    # -------------------------------
    # 1- describe() 
    # -------------------------------
    df.describe().show()
    """
    +-------+-----------------+----+
    |summary|           number|name|
    +-------+-----------------+----+
    |  count|                9|   9|
    |   mean|5.222222222222222|null|
    | stddev|3.073181485764296|null|
    |    min|                1|  z1|
    |    max|               10|  z9|
    +-------+-----------------+----+
    """
    # -------------------------------
    # 2- drop
    # -------------------------------
    triplets = [("alex", "Ames", 20),
                ("alex", "Sunnyvale", 30),
                ("alex", "Cupertino", 40),
                ("mary", "Ames", 35),
                ("mary", "Stanford", 45),
                ("mary", "Campbell", 55),
                ("jeff", "Ames", 60),
                ("jeff", "Sunnyvale", 70),
                ("jane", "Austin", 80)]

    df1 = spark.createDataFrame(triplets, ["name", "city", "age"])
    df1.show()
    df1_drop = df1.drop('city')
    df1_drop.show()

    # -------------------------------
    # 3- filter
    # -------------------------------
    df1.filter(df1.age > 50).show()
    df1.filter(df1.city.contains('me'))

    # -------------------------------
    # 4- join
    # -------------------------------
    ## 4.1 join_cross
    triplets2 = [("david", "software"),\
                ("david", "business"),\
                ("mary", "marketing"),\
                ("mary", "sales"),\
                ("jane", "genomics")]
                
    df_join = spark.createDataFrame(triplets2, ["name", "dept"])
    #        inner, cross, outer, full, full_outer, left,
    #        left_outer, right, right_outer, left_semi,
    #        and left_anti.
    df_joined = df1.join(df_join, df1.name == df_join.name, 'cross')
    df_joined.show()


    # -------------------------------
    # 5- sql
    # -------------------------------
    df1.createOrReplaceTempView('people')
    df3 = spark.sql('select name, city, age from people')
    df3.show()
    df1.groupBy(['name']).count().show()
    df5 = spark.sql(
        "SELECT name, count(*) as namecount FROM people GROUP BY name")
    df5.show()    
    spark.stop()
