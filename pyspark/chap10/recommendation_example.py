# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap10/recommendation_example.py
# # create date: 2020-02-17
# function: logistic_ 
# data: https://github.com/mahmoudparsian/pyspark-algorithms/edit/master/code/chap10/test.data


import sys, os
from pyspark.sql import SparkSession, SQLContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating 
from pyspark.mllib.recommendation import MatrixFactorizationModel


def create_rating(record_in):
    """
    return (userID, productID, rating)
    """
    tokens = record_in.split(',')  # 源github没加
    return Rating(int(tokens[0]), int(tokens[1]), float(tokens[2]))


if __name__ == '__main__':
    spark = SparkSession.builder.appName('logistic').getOrCreate()
    # 1- 载入数据
    input_path = r'E:\Work_My_Asset\pyspark_algorithms\chap1\test.data'
    rank = 10
    num_of_iter = 10

    # 2- 创建RDD
    dt = spark.sparkContext.textFile(input_path)

    # 3- 转换
    dt_deal = dt.map(create_rating)
    # 4- 模型构建
    model = ALS.train(dt_deal, rank, num_of_iter)

    # 5- 评估模型 
    dt_te = dt_deal.map(lambda r: (r[0], r[1]))
    pred_d = model.predictAll(dt_te).map(lambda r: ((r[0], r[1]), r[2]))
    rate_and_prd = dt_deal.map(lambda r:((r[0], r[1]), r[2])).join(pred_d)
    MSE = rate_and_prd.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))

    spark.stop()

