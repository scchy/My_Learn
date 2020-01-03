# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/dataframe_creation_with_explicit_schema.py
# create date: 2020-01-02
# function: dataframe creation 2
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap07/emps_with_header.txt

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, exp, rand,  expr, col
from pyspark.sql.functions import udf, sum, grouping_id
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField

from pyspark.sql import Row
import pandas as pd

def m_sqrt(n):
    return n * n



if __name__ == '__main__':
    spark = SparkSession.builder.appName('df_with_explicit_schema').getOrCreate()

    # ---------------------------
    # 7- df create with explicit schema
    # ---------------------------
    emps_schema = StructType([
        StructField('id', IntegerType(), True),
        StructField('name', StringType(), True),
        StructField("salary", IntegerType(), True),
        StructField("dept", StringType(), True)
    ])
    fil_name = r'E:\Work_My_Asset\pyspark_algorithms\chap1\emps_no_header.txt'

    emps_df = spark.read.csv(fil_name, schema = emps_schema)
    emps_df.show()
    emps_df.printSchema()
    emps_df.createOrReplaceTempView('emp_table')
    # SQL
    df_show1 = spark.sql('select * from emp_table where id > 1002')
    print("df_show1=", df_show1.show())
    df_show2 = spark.sql('select dept, count(*) from emp_table group by dept')
    print("df_show2=", df_show2.show())

    # ---------------------------
    # 8- crosstab
    # ---------------------------
    ## 类似excel透视  但只显示成对的频率
    emps_df.crosstab('name', 'dept').show()
    data = [(1, 1), (1, 2), (2, 1), (2, 1), (2, 3), (3, 2), (3, 3), (4, 4)]
    columns = ["key", "value"]
    df_cr = spark.createDataFrame(data, columns)
    df_cr.crosstab('key', 'value').show()

    # ---------------------------
    # 9- 删除重复行
    # ---------------------------
    data = [(100, "a", 1.0),
            (100, "a", 1.0),
            (200, "b", 2.0),
            (300, "c", 3.0),
            (300, "c", 3.0),
            (400, "d", 4.0)]

    columns = ("id", "code", "scale")
    df_dp = spark.createDataFrame(data, columns)
    df_dp.dropDuplicates().show()

    # ---------------------------
    # 10- groupby + agg & union
    # ---------------------------
    data = [
        ('Ames', 2006, 100),
        ('Ames', 2007, 200),
        ('Ames', 2008, 300),
        ('Sunnyvale', 2007, 10),
        ('Sunnyvale', 2008, 20),
        ('Sunnyvale', 2009, 30),
        ('Stanford', 2008, 90)
    ]
    columns = ("city", "year", "amount")
    sales = spark.createDataFrame(data, columns)
    sales_gb = sales.groupBy(['city', 'year']).agg(sum('amount').alias('amount'))
    print('group by df:')
    sales_gb.show()

    sales_gb_city = sales.groupBy('city').agg(sum('amount').alias('amount'))\
                    .select('city', lit(None).alias('year'), 'amount')
    
    sales_union = sales_gb.union(sales_gb_city)
    sales_union = sales_union.sort(
        sales_union.city.desc_nulls_last(), sales_union.year.asc_nulls_last())
    print('Group by with union:')
    sales_union.show()

    # ---------------------------
    # 11- rollup
    ## rollup 在group by 基础上增加组合的
    ## GROUPING_ID  单个表中储存多个级别的聚合时，该函数特别有用(例如使用rollup)
    ### 显示统计级别
    # ---------------------------
    """
    rollup 例子理解：
    1- groupBy(['dept', 'cat_flg']).agg(sum('cnt'))
    结果：
        dpt1	0	3635
        dpt2	0	2583
        dpt2	1	25 
    2- rollup('dept', 'cat_flg').agg(sum('cnt'), grouping_id().alias('gflg'))
    结果：
        (null)	(null)	6243  3 # 这条是 dept的单独汇总  (rollup 中的第一个特征的全部特征)
        dpt2	(null)	2608  1 # 这条是 dept2的单独汇总  (rollup 中的第一个特征的特征类型)
        dpt1	(null)	3635  1 # 这条是 dept1的单独汇总  (rollup 中的第一个特征的特征类型)
        ------- 下面三条一般group by就能出来
        dpt1	0	3635   0
        dpt2	0	2583   0
        dpt2	1	25     0
    """
    # 去掉全部的统计级数据
    with_rollup = sales.rollup('city', 'year')\
        .agg(sum('amount').alias('amount'), grouping_id().alias('gflg'))\
        .filter(col('gflg') != 3)
    with_rollup = with_rollup.sort(with_rollup.city.desc_nulls_last(), with_rollup.year.asc_nulls_last())\
        .select('city', 'year', 'amount', 'gflg')
    print("# with_rollup:")
    with_rollup.show()
    ## SQL
    sales.createOrReplaceTempView('sales_tbl')
    sales_SQL = spark.sql("""SELECT   city, year, sum(amount) as amount 
                                      ,GROUPING_ID(city, year) GFLG
                               FROM   sales_tbl
                           GROUP BY   ROLLUP(city, year)
                             HAVING   3 != GROUPING_ID(city, year) 
                           ORDER BY   city DESC NULLS LAST, year ASC NULLS LAST
                        """)
    print("# sales_SQL:")
    sales_SQL.show()

    spark.stop()

