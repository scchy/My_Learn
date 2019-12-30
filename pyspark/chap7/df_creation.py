# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/dataframe_creation_add_columns.py
# create date: 2019-12-30
# function: dataframe creation 
# data: 

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, exp, rand,  expr
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, DoubleType, StructType, StructField

zip_ = udf(
    lambda xs, ys: list(zip(xs, ys)),
    ArrayType(StructType([StructField('_1', StringType()), StructField('_2', DoubleType())]))
)






if __name__ == '__main__':
    spark = SparkSession.builder.appName('data_frame_creation').getOrCreate()
    data = [(100, "a", 3.0), (300, "b", 5.0)]
    col = ("x1", "x2", "x3")
    df = spark.createDataFrame(data, col)
    df.show()

    # ---------------------------
    # 1- 添加字段
    # ---------------------------
    ## 1.1 简单添加
    df_with = df.withColumn("x4", lit(0)).withColumn("x5", exp("x3"))
    df_with.show()
    ## 1.2 left join + rename | rand()
    other_data = [(100, "foo1"), (100, "foo2"), (200, "foo")]
    other_df = spark.createDataFrame(other_data, ("k", "v"))
    df_with_x6 = df_with.join(other_df, df_with.x1 == other_df.k, 'leftouter')\
                  .drop('k').withColumnRenamed('v', 'x6')
    df_with_x6.show()
    df_with_x6.withColumn("x8", rand()).show()

    # ---------------------------
    # 2- aggregate_multiple_columns
    # 在关联规则中可能用到
    # ---------------------------
    df = spark.sparkContext.parallelize([
        ("mary", "lemon", 2.00),
        ("adam", "grape", 1.22),
        ("adam", "carrot", 2.44),
        ("adam", "orange", 1.99),
        ("john", "tomato", 1.99),
        ("john", "carrot", 0.45),
        ("john", "banana", 1.29),
        ("bill", "apple", 0.99),
        ("bill", "taco", 2.59)
    ]).toDF(["name", "food", "price"])
    ### 2.1 文本agg collect_list
    df_coll = df.groupBy('name')\
                .agg(expr('collect_list(food) as food'), expr('collect_list(price) as price'))
    df_coll.show()
    df_coll_need = df_coll.withColumn('X', zip_(df_coll.food, df_coll.price))\
                    .drop('food').drop('price')
    df_coll_need.show(truncate=False)

    # ---------------------------
    # 3- 单列聚合 agg() + dict
    # ---------------------------
    df.groupBy("name").agg({'price': 'mean'}).show()
    df.groupBy("name").agg({'price': 'max'}).show()

    spark.stop()

