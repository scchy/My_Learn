[toc]


# 一、创建dataframe 
## 1.1 creation_from_collection
### 1.1.1 直接创建
```python
spark = SparkSession.builder.appName('Wod-Count-App').getOrCreate()

list_of_pairs = [("alex", 1), ("alex", 5), ("bob", 2),
                 ("bob", 40), ("jane", 60), ("mary", 700), ("adel", 800)]
df1 = spark.createDataFrame(list_of_pairs)
# 增加特征名
df1 = spark.createDataFrame(list_of_pairs, ["name", "value"])
```
### 1.1.2 结构内倒入数据
```python
schema = StructType([
    StructField('name', StringType(), True),
    StructField('age', IntegerType(), True)
])
df3 = spark.createDataFrame(list_of_pairs, schema)
```
### 1.1.3 从dict + func
```python
def covert_to_row(d: dict) -> Row:
    return Row(**OrderedDict(sorted(d.items())))
    

list_of_elements = [{"col1": "value11", "col2": "value12"}, {
    "col1": "value21", "col2": "value22"}, {"col1": "value31", "col2": "value32"}]
df4 = spark.sparkContext.parallelize(list_of_elements).map(covert_to_row).toDF()
```

## 1.2 creation_from_csv
```python
##  .option("header","true") 加载头 kv_with_header.txt
df = spark.read.format('csv').option('header', 'false')\
    .option('inferSchema', 'true').load(filname) 
df.show()

## 增加表头
df2 = df.selectExpr("_c0 as name", "_c1 as value")
df2.show()
```
## 1.3 creation_from_dict
```python
mydict = {'A': '1', 'B': '2', 'D': '8', 'E': '99'}
df = spark.createDataFrame(mydict.items(), ['key', 'value'])
df.show()

```
## 1.4 creation_from_dir
```python
# 用于相同结构分块存储的数据
df = spark.read.format('csv').option('header','false')\
    .option('inferSchema', 'true').load(dir_root+'/*.csv')
print("df.count = ",  df.count())
df.show()
```
## 1.5 creation_from_rdd
```python
list_of_tuples = [('alex','Sunnyvale', 25), ('mary', 'Cupertino', 22), ('jane', 'Ames', 20), ('bob', 'Stanford', 26)]
rdd = spark.sparkContext.parallelize(list_of_tuples)
people = rdd.map(lambda x: Row(name=x[0], city=x[1], age=int(x[2])))

df = spark.createDataFrame(people)
df.show()
```

# 二、创建rdd

## 2.1 from collection
```python
list_of_strs = ['alx', 'bob', 'jane', 'mary', 'adel']
rdd1 = spark.sparkContext.parallelize(list_of_strs)
## list pairs
list_of_pair = [("alex", 1), ("alex", 3), ("alex", 9), ("alex", 10), ("bob", 4), ("bob", 8)]
rdd2 = spark.sparkContext.parallelize(list_of_pair)
```
## 2.2 from dict 
```python
d = {"key1":"value1","key2":"value2","key3":"value3"} 
rdd_f_dict = spark.sparkContext.parallelize(d.items())
```
## 2.3 from df 
```python
list_of_pairs = [("alex", 1), ("alex", 5), ("bob", 2), ("bob", 40), ("jane", 60), ("mary", 700), ("adel", 800) ]
df = spark.createDataFrame(list_of_pairs)
df.show()
# Convert an existing DataFrame to an RDD
rdd = df.rdd
print("rdd.collect() = ",  rdd.collect())
```
## 2.4 from dir 
```python
rdd_dir = spark.sparkContext.textFile(dir_name+'/*.txt')
```
