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
from pyspark.sql.types import ArrayType, StringType, DoubleType, StructType, StructField, LongType

from pyspark.sql import Row
import pandas as pd

def m_sqrt(n):
    return n * n


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

    
    # ---------------------------
    # 4- udf
    # ---------------------------
    data = [('alex', 5), ('jane', 7), ('bob', 9)]
    df_udf = spark.createDataFrame(data, ['name', 'age'])
    sqrt_udf = udf(m_sqrt, LongType())
    df_udfed = df_udf.select('name', 'age', sqrt_udf('age').alias('age_sqrt'))
    df_udfed.show()

    # ---------------------------
    # 5- df from pandas 
    # ---------------------------
    pd_df = pd.DataFrame(
        data={
            'integers': [2, 5, 7, 8, 9],
            'floats': [1.2, -2.0, 1.5, 2.7, 3.6],
            'int_arrays': [[6], [1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12]]
        }
    )
    spark_df = spark.createDataFrame(pd_df)
    spark_df.show()
    # 转回pandas 
    pandas_df = spark_df.toPandas()
    print("pandas_df = \n", pandas_df)
    ## Orderby 
    spark_df.orderBy(spark_df.integers.desc()).show()
    spark_df.orderBy(spark_df.floats, spark_df.integers.desc()).show()
    
    # ---------------------------
    # 6- df from Row
    # ---------------------------
    dpt1 = Row(id='100', name='Computer Science')
    dpt2 = Row(id='200', name='Mechanical Engineering')
    dpt3 = Row(id='300', name='Music')
    dpt4 = Row(id='400', name='Sports')
    dpt5 = Row(id='500', name='Biology')

    Employee = Row("first_name", "last_name", "email", "salary")
    employee1 = Employee('alex', 'smith', 'alex@berkeley.edu', 110000)
    employee2 = Employee('jane', 'goldman', 'jane@stanford.edu', 120000)
    employee3 = Employee('matei', None, 'matei@yahoo.com', 140000)
    employee4 = Employee(None, 'eastwood', 'jimmy@berkeley.edu', 160000)
    employee5 = Employee('betty', 'ford', 'betty@gmail.com', 130000)

    dpt_emp1 = Row(department=dpt1, employees=[employee1, employee2, employee5])
    dpt_emp2 = Row(department=dpt2, employees=[employee3, employee4])
    dpt_emp3 = Row(department=dpt3, employees=[employee1, employee4])
    dpt_emp4 = Row(department=dpt4, employees=[employee2, employee3])
    dpt_emp5 = Row(department=dpt5, employees=[employee5])
    # 查看
    print("dpt1=", dpt1)
    print("employee1=", employee1)
    print("dpt_emp1=", dpt_emp1)
    # 创建df
    dpt_emt_list = [eval('dpt_emp{}'.format(i)) for i in range(1, 6)]
    df_row = spark.createDataFrame(dpt_emt_list)
    df_row.show(truncate=False)
    
    spark.stop()
