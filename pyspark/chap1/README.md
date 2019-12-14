@[toc]

# 一、基本操作
```python
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import * # 有struct 和 dt_type
import pyspark.sql.functions as sql_func
```
## 1.1 创建spark连接
### 1.1.1 SparkSession
```python
spark = SparkSession.builder.appName("session_name").getOrCreate()
```
### 1.1.2 Sparkconf
```python
from pyspark import SparkConf 

conf = SparkConf().setAppName('WordCount')
conf.set('spark.executor.memory', '500M')
conf.set('spark.cores.max', 4)

## 读取数据方法同 sparksession 
## 如下
rdd = sc.textFile(fil_name)
```


## 1.2 数据加载
### 1.2.1 载入json
```python
df = spark.read.json(json_path)
df.show()

```
### 1.2.2 载入文本
```python
results = spark.sparkContext.textFile(fil_name)
```

### 1.2.3 载入csv
**需要先搭建框架在载入数据**
```python
schema = StructType([
    StructField('name', StringType()),
    StructField('city', StringType()),
    StructField('age', DoubleType())
])

df = spark.read.schema(schema).format('csv')\
    .option('header', 'false')\
    .option('inferSchema', 'true')\
    .load(fil_name)

df.show()
```
## 1.3 一般操作
### 1.3.1 json等有表头的数据
```python
## select
df.select('name').show()
df.select(df['name'], df['age'] + 1).show()

## where
df.filter(df['age'] > 23).show()
## groupBy
df.groupBy('age').count().show()


# df 命名成people
df.createOrReplaceTempView('people') 
sql_df = spark.sql('select * from people')
sql_df.show()

# Register the df as a globeltemporary view
df.createGlobalTempView('people')
spark.sql('select * from global_temp.people').show()

spark.stop()
```
### 1.3.2 rdd操作
```python
def compute_stats(num_dt):
    avg = stat.mean(num_dt)
    median = stat.median(num_dt)
    std = stat.stdev(num_dt)
    return avg, median, std

def create_pair(record):
    tokens = record.split(',')
    url_address = tokens[0]
    frequency = int(tokens[1])
    return (url_address, frequency)
```
#### 1.3.2.1 简单操作
```python
# where
resf = results.filter(lambda record: len(record) > 5)
# map 映射
resf = resf.map(create_pair)
# groupby 然后计算数值 映射方法是 compute_stats
resf = resf.groupByKey().mapValues(compute_stats)

resf.collect()
spark.stop()

# 同样可以做累加
# reduceByKey(lambda x, y: x + y) 
```
#### 1.3.2.2 排序
```python
records = spark.sparkContext.textFile(fil_name)
print("展平>>增加一列>>排序")
sorted_cnt = records.flatMap(lambda rec: rec.split(' '))\
    .map(lambda n: (int(n), 1)).sortByKey()
print(sorted_cnt.collect())
output = sorted_cnt.collect()
```

### 1.3.3 结构框架下的CSV数据
- 链式操作
```python
average_method1 = df.groupBy('city').agg(sql_func.avg('age').alias('average'))
average_method1.show()
```
- spark.sql 操作  
需要创建视图
```python
df.createOrReplaceTempView('df_tbl')
average_method2 = spark.sql("select city, avg(age) avg_age from df_tbl group by city")
average_method2.show()
spark.stop()
```
