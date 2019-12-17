[toc]
# 一、聚合
## 1.1 groupByKey()
```python
res = records.map(create_pair).groupByKey().mapValues(lambda v: sum(v))
```
## 1.2 reduceByKey()
```python
res = records.map(create_pair).reduceByKey(lambda a, b: a+b)
```

# 二、筛选&sort
```python
words = record.filter(lambda x: len(x) > 0).flatMap(lambda line: line.lower().split(" "))
```
```python
words_count_group = words.map(lambda x: (x, 1)).groupByKey().mapValues(lambda x : sum(x))

# 增加sort 
words_count_sort = words_count.sortBy(lambda x: x[1], ascending = False)

# 增加filter
filtered = words_count.filter(lambda x: x[1] > 2)
```

