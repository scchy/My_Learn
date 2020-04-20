import tensorflow as tf

# 1- tf.nn.bias_add 添加偏置
#----------------------------------
value_a = tf.constant([[1,1], [2,2], [3,3]], dtype=tf.float32)
bias_x = tf.constant([1, -1], dtype=tf.float32)
# 列的维度添加偏置
"""
>>> tf.nn.bias_add(value_a, bias_x)
<tf.Tensor: id=2, shape=(3, 2), dtype=float32, numpy=
array([[2., 0.],
       [3., 1.],
       [4., 2.]], dtype=float32)>
"""
