[toc]

# 一、DataFrame一些操作
> 一些操作均需要在 `pyspark.sql.functions` 中加载 

- 数据
```
import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, exp, rand,  expr
from pyspark.sql.functions import udf, sum, grouping_id
from pyspark.sql.types import ArrayType, StringType, DoubleType, StructType, StructField, LongType

from pyspark.sql import Row
import pandas as pd

spark = SparkSession.builder.appName('data_frame_creation').getOrCreate()
data = [(100, "a", 3.0), (300, "b", 5.0)]
col = ("x1", "x2", "x3")
df = spark.createDataFrame(data, col)

other_data = [(100, "foo1"), (100, "foo2"), (200, "foo")]
other_df = spark.createDataFrame(other_data, ("k", "v"))
```

## 1.1 添加列
1.简单添加
```python
df_with = df.withColumn("x4", lit(0)).withColumn("x5", exp("x3"))
```
2.关联添加
```
df_with_x6 = df_with.join(other_df, df_with.x1 == other_df.k, 'leftouter')\
              .drop('k').withColumnRenamed('v', 'x6')

df_with_x6.withColumn("x8", rand()).show()
```

## 1.2 udf
> 定义出一个`udf`功能其实就是类似函数， 但是需要指明结构  
> 定以后可以和 sql.functions 一样用

```python
def m_sqrt(n):
    return n * n
    
data = [('alex', 5), ('jane', 7), ('bob', 9)]
df_udf = spark.createDataFrame(data, ['name', 'age'])

# udf (func, struct)
sqrt_udf = udf(m_sqrt, LongType())
df_udfed = df_udf.select('name', 'age', sqrt_udf('age').alias('age_sqrt'))
df_udfed.show()

```

## 1.3 多行聚合
> 合并行 类似 msql `group_concat(list, separeator ',')`

```python
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

df_coll = df.groupBy('name')\
                .agg(expr('collect_list(food) as food'), expr('collect_list(price) as price'))
```
- 增加 udf 再聚合
```python
from pyspark.sql.functions import udf

zip_ = udf(
    lambda xs, ys: list(zip(xs, ys)),
    ArrayType(StructType([StructField('_1', StringType()), StructField('_2', DoubleType())]))
)

df_coll_need = df_coll.withColumn('X', zip_(df_coll.food, df_coll.price))\
                .drop('food').drop('price')
df_coll_need.show(truncate=False)

```

## 1.4 单行聚合
```python
df.groupBy("name").agg({'price': 'mean'}).show()
df.groupBy("name").agg({'price': 'max'}).show()
```

## 1.5 从Row结构到DataFrame
```python
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

# 行嵌行
dpt_emp1 = Row(department=dpt1, employees=[employee1, employee2, employee5])
dpt_emp2 = Row(department=dpt2, employees=[employee3, employee4])
dpt_emp3 = Row(department=dpt3, employees=[employee1, employee4])
dpt_emp4 = Row(department=dpt4, employees=[employee2, employee3])
dpt_emp5 = Row(department=dpt5, employees=[employee5])

# 创建df
dpt_emt_list = [eval('dpt_emp{}'.format(i)) for i in range(1, 6)]
df_row = spark.createDataFrame(dpt_emt_list)
df_row.show(truncate=False)
```

## 1.6 交叉频率表(`crosstab`)
- 类似excel透视  但只显示成对的频率
```python
data = [(1, 1), (1, 2), (2, 1), (2, 1), (2, 3), (3, 2), (3, 3), (4, 4)]
columns = ["key", "value"]
df_cr = spark.createDataFrame(data, columns)
df_cr.crosstab('key', 'value').show()
```

## 1.7 删除重复行(`dropDuplicates`)
```python
data = [(100, "a", 1.0),
        (100, "a", 1.0),
        (200, "b", 2.0),
        (300, "c", 3.0),
        (300, "c", 3.0),
        (400, "d", 4.0)]

columns = ("id", "code", "scale")
df_dp = spark.createDataFrame(data, columns)
df_dp.dropDuplicates().show()
```

## <font color='red'> 1.8 gruopby组合(`rollup`&`GROUPING_ID`) </font>
<font color = steelblue>

rollup 例子理解(数值编的)：  
1- `groupBy(['dept', 'cat_flg']).agg(sum('cnt')) `   
结果：
dept|cat_flg| cnt
-|-|-
dpt1|0|3635
dpt2|0|2583
dpt2|1|25 
2- `rollup('dept', 'cat_flg').agg(sum('cnt'), grouping_id().alias('gflg'))`  
结果：  
dept|cat_flg| cnt|gflg|备注
-|-|-|-|-
(null)|(null)|6243|3|dept的单独汇总(rollup中的第一个特征的全部特征)
dpt2|(null)|2608|1|dept2的单独汇总(rollup中的第一个特征的特征类型)
dpt1|(null)|3635|1|dept1的单独汇总(rollup中的第一个特征的特征类型)
dpt1|0|3635|0|group by就能出来
dpt2|0|2583|0|group by就能出来
dpt2|1|25|0|group by就能出来
        
</font>

- spark中rollup

```python
# 去掉全部的统计级数据
with_rollup = sales.rollup('city', 'year')\
    .agg(sum('amount').alias('amount'), grouping_id().alias('gflg'))\
    .filter(col('gflg') != 3)

# 排序
with_rollup = with_rollup.sort(with_rollup.city.desc_nulls_last(), with_rollup.year.asc_nulls_last())\
    .select('city', 'year', 'amount', 'gflg')
print("# with_rollup:")
with_rollup.show()
```

- sparl.sql 实现
```python
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
```

# 二、简单数值型数据探索
## 2.1 summary
```python
df.summary().select('life_exp').show()
```

## 2.2 近似百分位 快速求解(`approxQuantile`)
```python
quantileProbs = [0.01, 0.25, 0.5, 0.75, 0.99]
relError = 0.01
explore_dt = df.approxQuantile("life_exp", quantileProbs, relError)
```

# 三、分区导出导入
> 导出需要在lunix 下 文件命名 windows 不允许

```python
df = spark.read.option('delimiter', ',').option('inferSchema', 'true').csv(fil_name).toDF(
    'customer_id', 'year', 'transaction_id', 'transaction_value')
df.show(truncate=False)

# 分区输出
# partition data by 'customer_id', and then by 'year'
# 需要在lunix 下 文件命名 windows 不允许
part_path = r'E:\Work_My_Asset\pyspark_algorithms\chap1\part' 

df.repartition('customer_id', 'year')\
    .write.partitionBy('customer_id', 'year')\
    .parquet(part_path)

# 导入
df2 = spark.read.parquet(part_path)
df2.show(truncate=False)
```

[参考: https://github.com/mahmoudparsian/pyspark-algorithms](https://github.com/mahmoudparsian/pyspark-algorithms/tree/master/code/chap05)
