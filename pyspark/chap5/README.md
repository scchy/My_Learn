[toc]

# 一、聚合方法
> 大型数据集中，为了减少shuffle的数据量，相对于groupByKey来说，使用reduceByKey、combineByKey以及foldByKey 会是更好的选择
- 数据

```python
def create_pair(t3):
    return (t3[0], int(t3[2]))

def create_pair_sumcnt(t3):
    name = t3[0]
    number = int(t3[2])
    return (name, (number, 1))

spark = SparkSession.builder.appName('avg_by_key').getOrCreate()
list_of_tuples = [('alex', 'Sunnyvale', 25),
                  ('alex', 'Sunnyvale', 33),
                  ('alex', 'Sunnyvale', 45),
                  ('alex', 'Sunnyvale', 63),
                  ('mary', 'Ames', 22),
                  ('mary', 'Cupertino', 66),
                  ('mary', 'Ames', 20),
                  ('bob', 'Ames', 26)]
rdd = spark.sparkContext.parallelize(list_of_tuples)
rdd_pair = rdd.map(lambda t: create_pair(t))

rdd_pair1 = rdd.map(lambda x: create_pair_sumcnt(x))
```
## 1.1 aggregateByKey
1. 预设初始值 `zero_tuple = (0, 0)`
2. Comb操作，每个数值变量变化 例如 2 -> (2, 1)
3. Reduce 操作
```python
sum_count = rdd_pair.aggregateByKey(
    (0 , 0),
    lambda zero_tuple, values_: (zero_tuple[0] + values_, zero_tuple[1] + 1),
    lambda tup0, tup1: (tup0[0] + tup1[0], tup0[1]+tup1[1])  # reduce
)
```

## 1.2 acombineByKey
- 源码
```python
def combineByKey[C](
  createCombiner: V => C,
  mergeValue: (C, V) => C,
  mergeCombiners: (C, C) => C): RDD[(K, C)] = self.withScope {
combineByKeyWithClassTag(createCombiner, mergeValue, mergeCombiners)(null)
}
```
> 　　　　1、createCombiner:V=>C　　分组内的创建组合的函数。通俗点将就是对读进来的数据进行初始化，其把当前的值作为参数，可以对该值做一些转换操作，<font color = red>转换为我们想要的数据格式</font>  
>
>2、mergeValue:(C,V)=>C　　该函数主要是分区内的合并函数，作用在每一个分区内部。<font color = red>其功能主要是将V合并到之前(createCombiner)的元素C上</font> ,注意，<font color=red>这里的C指的是上一函数转换之后的数据格式</font>，而<font color=red>这里的V指的是原始数据</font>(上一函数为转换之前的)
>
>3、mergeCombiners:(C,C)=>R　　该函数主要是进行多分取合并，此时是将两个C合并为一个C，例如两个C:(Int)进行相加之后得到一个R:(Int)。<font color=red>相当于reduce(底层调用相同)。</font>
>
> <font color=orange>combineByKey与reduceByKey的区别是：combineByKey可以返回与输入数据类型不一样的输出。</font>

- 例子
```python
sum_count1 = rdd_pair.combineByKey(
    lambda values_: (values_, 1), # createCombiner
    lambda combined, values_: (combined[0] + values_, combined[1] + 1), # 在combined中叠加
    lambda values_, cnts: (values_[0] + cnts[0], values_[1]+cnts[1])  # reduce
)
```

## 1.3 foldByKey
```python
sum_count2 = rdd_pair1.foldByKey(
    (0, 0),  # zero_value = (0, 0) = (sum, count)
    lambda x, y : (x[0] + y[0], x[1] + y[1])
)
```

## 1.4 groupby 
```python
sum_count3 = rdd_pair.groupByKey().mapValues(lambda x : (sum(list(x)), len(list(x))))
# 或者分步减少计算
#  sum_count3 = rdd_pair.groupByKey().mapValues(lambda x : list(x) )
#  avg = sum_count3.map(lambda x : float(sum(x)) / float(len(x)))
```

## 1.5 reduceByKey 
```python
sum_count4 = rdd_pair1.reduceByKey(
    lambda fs, nd: (fs[0] + nd[1], fs[0] + nd[1])
    )
```


# 二、DataFrame操作
- 数据

```python
spark = SparkSession.builder.appName('df_action').getOrCreate()
pairs = [(10,"z1"), (1,"z2"), (2,"z3"), (9,"z4"), (3,"z5"), (4,"z6"), (5,"z7"), (6,"z8"), (7,"z9")]
df = spark.createDataFrame(pairs, ['number', 'name'])

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
```

## 2.1 describe
```python
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
```

## 2.2 drop
```python
df1_drop = df1.drop('city')
df1_drop.show()
```

## 2.3 join
方法和pandas的merge相似
```python
df1.join(df_join, df1.name == df_join.name, 'cross')
# join 方式
#        inner, cross, outer, full, full_outer, left,
#        left_outer, right, right_outer, left_semi,
#        and left_anti
```

## 2.4 sql
方法和sql一样
```python
df.createOrReplaceTempView('df_name')
spark.sql('select name, count(1) from df_name
group by name').show()
```


## 2.5 withColumn 增加列
```python
df_withcol = df1.withColumn('age2', df1.age + 2)
df_withcol.show()

# 同样可以用SQL方法实现
df.createOrReplaceTempView('df_name')
spark.sql('select *, age + 2 as agw2 from df_name').show()
```

# 三、RDD操作
- 数据
```python
def tokenize(record):
    tokens = record.split(' ')
    mylist = []
    for word in tokens:
        if len(word) > 2:
            mylist.append(word)
    return mylist

b = [('p', 50), ('x', 60), ('y', 70), ('z', 80) ]
a = [('a', 2), ('b', 3), ('c', 4)]
rdd_a = spark.sparkContext.parallelize(a)
rdd_b = spark.sparkContext.parallelize(b)
```

## 3.1 cartesian
卡迪尔叠加
```python
cart = rdd_a.cartesian(rdd_b)
```
## 3.2 filter
```python
rdd_f = rdd_b.filter(lambda x: x[1] > 55)
```

## 3.3 flatmap
```python
list_of_strings = ['of', 'a fox jumped',
                   'fox jumped of fence', 'a foxy fox jumped high']
rdd_flat = spark.sparkContext.parallelize(list_of_strings)
rdd_flated = rdd_flat.flatMap(lambda rec: tokenize(rec))
```

## 3.4 join
```python
source_pairs = [(1, "u"), (1, "v"), (2, "a"), (3, "b"), (4, "z1")]
source = spark.sparkContext.parallelize(source_pairs)

other_pairs = [(1, "x"), (1, "y"), (2, "c"), (2, "d"), (3, "m"), (8, "z2")]
other = spark.sparkContext.parallelize(other_pairs)
joined = source.join(other)
# [(1, ('u', 'x')), (1, ('u', 'y')), (1, ('v', 'x')), (1, ('v', 'y')), (2, ('a', 'c')), (2, ('a', 'd')), (3, ('b', 'm'))]
```

## <font color = red>3.5 mapPartitions</font>
### 3.5.1 mapPartitions例子
```python
numbers = ["10,20,3,4",
           "3,5,6,30,7,8",
           "4,5,6,7,8",
           "3,9,10,11,12",
           "6,7,13",
           "5,6,7,12",
           "5,6,7,8,9,10",
           "11,12,13,14,15,16,17"]
# len(numbers) == 8

rdd_em = spark.sparkContext.parallelize(numbers, 10)
min_max_count_rdd = rdd_em.mapPartitions(min_max_count)
```
### 3.5.2 每个分区内的iter处理(含空分区 和 分区内多iter)
```python
def min_max_count(iterator: iter) -> list:
    try:
        n = 0
        for ite_i in iterator:
            n += 1
            numbers = ite_i.split(",")
            # convert strings to integers
            numbers = list(map(int, numbers))
            print(numbers)
            if n == 1 :
                local_min = min(numbers)
                local_max = max(numbers)
                local_count = len(numbers)
            else: # 处理partition小于时(含多个iter)
                if local_min > min(numbers):
                    local_min = min(numbers)
                if local_max < min(numbers):
                    local_max = min(numbers)
                local_count += len(numbers)
        return [(local_min, local_max, local_count)]
    except : # 处理partition 为空时
        # WHERE min > max to filter it out later
        return [(1, -1, 0)]
```

## 3.6 sortBy
- sortByKey 根据主键
- sortBy 用lambda 自定
```python
pairs = [(10, "z1"), (1, "z2"), (2, "z3"), (9, "z4"), (3, "z5"), (4, "z6"), (5, "z7"), (6, "z8"), (7, "z9")]
rdd_st = spark.sparkContext.parallelize(pairs)
print('值排序，倒序', rdd_st.sortBy(lambda x:x[1], ascending = False).collect())
print('主键排序，倒序', rdd_st.sortByKey(ascending=False).collect())

```
## 3.7 takeOrdered
```python
bottom3 = rdd_st.takeOrdered(3, key=lambda x: -x[0]) # return list
print("bottom3 = ", bottom3)
```

[参考: https://github.com/mahmoudparsian/pyspark-algorithms](https://github.com/mahmoudparsian/pyspark-algorithms/tree/master/code/chap05)
