# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import time
from tensorflow.python.platform import gfile

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

data_root = pathlib.Path("G:/works/2020/py-tensorflow/numberdata/data")
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

# 列出可用的标签
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

# 为每个标签分配索引
label_to_index = dict((name, index) for index, name in enumerate(label_names))


# 创建一个列表，包含每个文件的标签索引
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])
print(all_image_paths)
print(label_names)
print(label_to_index)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0
    return image

def load_and_preprocess_image(path):
  print(path)
  image = tf.io.read_file(path)
  return preprocess_image(image)

def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return image_rel


# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

def change_range(image,label):
  return 2*image-1, label





image_path = all_image_paths[0]
label = all_image_labels[0]
# plt.imshow(load_and_preprocess_image(image_path))
# plt.grid(False)
# plt.xlabel("x")
# plt.title(label_names[label].title())
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
print(image_ds)
# #image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


# ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
# image_label_ds = ds.map(load_and_preprocess_from_path_label)

# # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# # 被充分打乱。
# image_count = len(all_image_paths)
# # ds = image_label_ds.shuffle(buffer_size=image_count)
# # ds = ds.repeat()
# # ds = ds.batch(BATCH_SIZE)
# ds = image_label_ds.cache(filename='./cache.tf-data')
# ds = ds.apply(
#   tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
# ds = ds.batch(BATCH_SIZE).prefetch(1)

# # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
# ds = ds.prefetch(buffer_size=AUTOTUNE)

# mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
# mobile_net.trainable=False

# keras_ds = ds.map(change_range)

# # 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
# image_batch, label_batch = next(iter(keras_ds))

# feature_map_batch = mobile_net(image_batch)

# model = tf.keras.Sequential([
#   mobile_net,
#   tf.keras.layers.GlobalAveragePooling2D(),
#   tf.keras.layers.Dense(len(label_names), activation = 'softmax')])

# logit_batch = model(image_batch).numpy()

# print("min logit:", logit_batch.min())
# print("max logit:", logit_batch.max())
# print()

# print("Shape:", logit_batch.shape)


# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='sparse_categorical_crossentropy',
#               metrics=["accuracy"])

# model.summary()

# steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
# default_timeit_steps = 2*steps_per_epoch+1

# def timeit(ds, steps=default_timeit_steps):
#   overall_start = time.time()
#   # 在开始计时之前
#   # 取得单个 batch 来填充 pipeline（管道）（填充随机缓冲区）
#   it = iter(ds.take(steps+1))
#   next(it)

#   start = time.time()
#   for i,(images,labels) in enumerate(it):
#     if i%10 == 0:
#       print('.',end='')
#   print()
#   end = time.time()

#   duration = end-start
#   print("{} batches: {} s".format(steps, duration))
#   print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
#   print("Total time: {}s".format(end-overall_start))


# print("steps_per_epoch:", steps_per_epoch)
# timeit(ds)
# model.fit(ds, epochs=1, steps_per_epoch=1500)
# model.save('G:/works/2020/py-tensorflow/numberdata/saved_model/my_model')


# image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
# tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
# tfrec.write(image_ds)

# image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)

# ds = tf.data.Dataset.zip((image_ds, label_ds))
# ds = ds.apply(
#   tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
# ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# timeit(ds)

# paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# image_ds = paths_ds.map(load_and_preprocess_image)
# ds = image_ds.map(tf.io.serialize_tensor)
# tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
# tfrec.write(ds)


