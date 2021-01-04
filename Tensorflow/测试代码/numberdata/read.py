# -*- coding: utf-8 -*-
# import os
# import tensorflow as tf
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# from tensorflow.python.platform import gfile

import tensorflow as tf
import numpy as np



# tfrecords_file="G:/works/2020/py-tensorflow/numberdata/data/train.tfrecords"

# filenames = [tfrecords_file]
# raw_dataset = tf.data.TFRecordDataset(filenames)
# for raw_record in raw_dataset.take(10):
#   print(repr(raw_record))


#   # Create a description of the features.
# feature_description = {
#     'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
# }

# def _parse_function(example_proto):
#   # Parse the input `tf.Example` proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, feature_description)


#   parsed_dataset = raw_dataset.map(_parse_function)

#   for parsed_record in parsed_dataset.take(10):
#       print(repr(parsed_record))

# 实现线性回归
# x = [1,1]
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1, input_shape=(192, 192)))
# model.summary()

# model.compile(optimizer=tf.keras.optimizers.Adam, loss='mse',  metrics=["accuracy"])
# model.fit(x, epochs=1, steps_per_epoch=1500)


random_float = tf.random.uniform(shape=())
zero_vector = tf.zeros(shape=(2))

A = tf.constant([[1,2], [3,4]])
B = tf.constant([[5,6], [7,8]])

print(tf.add(A,B), tf.matmul(A,B))

x = tf.Variable(initial_value=3.0)
with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad = tape.gradient(y, x)
print([y, y_grad])



x_row = np.array([2013,2014,2015,2016,2017], dtype=np.float32)
y_row = np.array([12000,14000,15000,16500,17500], dtype=np.float32)

X = [x_row - x_row.min()] / (x_row.max() - x_row.min())
y = [y_row - y_row.min()] / (y_row.max() - y_row.min())

X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))


print(a,b)

tf.config.experimental_functions_run_eagerly(True)