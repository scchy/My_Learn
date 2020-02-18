# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap11/breadth_first_search_example.py
# # create date: 2020-02-17
# function: logistic_ 
# data: https://github.com/mahmoudparsian/pyspark-algorithms/edit/master/code/chap10/test.data

# 广度优先搜索
# 1- Builds a graph using GraphFrame package
# 2- Applies BFS find the shortest paths from one vectex to another



import sys, os
from pyspark.sql import SparkSession, SQLContext
from graphframes import GraphFrame



if __name__ == '__main__':
    spark = SparkSession.builder.appName('BFS').getOrCreate()
    # 1- 载入数据
    vertices = [("a", "Alice", 30),
                ("b", "Bob", 31),
                ("c", "Charlie", 32),
                ("d", "David", 23),
                ("e", "Emma", 24),
                ("f", "Frank", 26)]
    v = spark.createDataFrame(vertices, ["id", "name", "age"])

    edges = [("a", "b", "follow"),
             ("b", "c", "follow"),
             ("c", "d", "follow"),
             ("d", "e", "follow"),
             ("b", "e", "follow"),
             ("c", "e", "follow"),
             ("e", "f", "follow")]
    e = spark.createDataFrame(edges, ["src", "dst", "relationship"])
    e.show()
    # 2- 创建图
    ## 报错修正(少jars包)：https://blog.csdn.net/qq_15098623/article/details/91533349
    graph = GraphFrame(v, e)
    print("graph=", graph)

    # 3- 广度搜索
    paths = graph.bfs("name = 'Alice'", "age < 27")
    paths.show()
    

    spark.stop()

