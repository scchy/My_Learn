# python 3.6
# author(learning): Scc_hy
# original url: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap09/logistic_regression_builder.py
# # create date: 2020-02-17
# function: logistic_ 
# data: https://github.com/mahmoudparsian/pyspark-algorithms/blob/master/code/chap09/training_emails_spam.txt


import sys, os
from pyspark.sql import SparkSession, SQLContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification  import LogisticRegressionWithSGD

def createLabeledPoint(label, email, tf):
    tokens = email.split() # 源github没加
    features = tf.transform(tokens)
    return LabeledPoint(label, features)


def classify(email, tf, model):
    """
    用以预测新的email是否欺诈
    例如:
        tf = HashingTF(numFeatures=128)
        classify(new_eml, tf, model)
    """
    # tokenize an email into words
    tokens = email.split()
    # create features for a given email
    features = tf.transform(tokens)
    # classify email into "spam" (class 1) or "non-spam" (class 0)
    return model.predict(features)


if __name__ == '__main__':
    spark = SparkSession.builder.appName('logistic').getOrCreate()
    # 1- 载入数据
    tr1_fil = r'E:\Work_My_Asset\pyspark_algorithms\chap1\training_emails_spam.txt'
    tr0_fil = r'E:\Work_My_Asset\pyspark_algorithms\chap1\training_emails_nospam.txt'
    tr1_rdd = spark.sparkContext.textFile(tr1_fil)
    tr0_rdd = spark.sparkContext.textFile(tr0_fil)

    # 2- 创建HTF 分词矩阵
    ## tf is an instance of HashingTF
    ##, which can hold up to 128 features 
    FEATURES_HTF = HashingTF(numFeatures=128)

    # 3- 增加标签
    tr1_rdd_labeled = tr1_rdd.map(
        lambda email: createLabeledPoint(1, email, FEATURES_HTF))

    tr0_rdd_labeled = tr0_rdd.map(
        lambda email: createLabeledPoint(0, email, FEATURES_HTF))
    tr_data = tr1_rdd_labeled.union(tr0_rdd_labeled)

    # 4- 模型构建
    LR_model = LogisticRegressionWithSGD.train(tr_data)

    # 5- save The built model
    # saved_model_path = r'E: \Work_My_Asset\pyspark_algorithms\chap1\model'
    # LR_model.save(spark.sparkContext, saved_model_path)

    #====================================
    # Calculate the accuracy of the model.
    #====================================
    tr_7, te_3 = tr_data.randomSplit((0.7, 0.3))
    print("te_3.count()=", te_3.count())
    print("te_3.collect()=", te_3.collect())

    pred_te = te_3.map(lambda x: (LR_model.predict(x.features), x.label))
    print("predict label count:", pred_te.count())
    accu = 1.0 * pred_te.filter(lambda x: float(x[0]) == float(x[1])).count() / te_3.count()
    print('accuracy = ', accu)



    spark.stop()

