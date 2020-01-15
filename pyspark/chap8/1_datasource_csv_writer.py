# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap08/datasource_csv_writer.py
# create date: 2020-01-10
# function: datasource_csv_write
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/customers.txt

import sys, os
from pyspark.sql import SparkSession
from os.path import isfile, join


def debug_file(input_path):
    f = open(input_path, 'r')
    file_contents = f.read()
    print ("file_contents = \n" + file_contents)
    f.close()


def dump_directory(dir):
    print("output dir name: ", dir)
    #contents of the current directory    
    dir_listing = os.listdir(dir) 
    print("dir_listing: ", dir_listing)
    for path in dir_listing:
        if path.startswith("part"):
            fullpath = join(dir, path)
            if isfile(fullpath) == True :          
                print("output file name: ", fullpath)
                debug_file(fullpath)



if __name__ == '__main__':
    spark = SparkSession.builder.appName('datasource_csv_writer').getOrCreate()
    output_csv_file_path = r'E:\Work_My_Asset\pyspark_algorithms\chap1\people_csv'

    column_names = ["name", "city", "age"]
    people = spark.createDataFrame([\
        ("Alex", "Ames", 50),\
        ("Alex", "Sunnyvale", 51),\
        ("Alex", "Stanford", 52),\
        ("Gandalf", "Cupertino", 60),\
        ("Thorin", "Sunnyvale", 95),\
        ("Max", "Ames", 55),\
        ("George", "Cupertino", 60),\
        ("Terry", "Sunnyvale", 95),\
        ("Betty", "Ames", 78),\
        ("Brian", "Stanford", 77)], column_names)
    people.write.csv(output_csv_file_path)
    dump_directory(output_csv_file_path) # 需要下载had.ll
    people.show(10, truncate=False)
    print("people.collect() = " , people.collect())

    spark.stop()

