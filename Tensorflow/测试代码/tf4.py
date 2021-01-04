# -*- coding: UTF-8 -*-
import tensorflow as tf
import pathlib
import random
import os
import IPython.display as display
import matplotlib.pyplot as plt

data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)

# for item in data_root.iterdir():
#     print(item)


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
# print(image_count, all_image_paths[:10])

attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    k = str(image_rel).replace('\\', '/')
    return "Image (CC BY 2.0) " + ' - '.join(attributions[k].split(' - ')[:-1])


# for n in range(3):
#   image_path = random.choice(all_image_paths)
#   display.display(display.Image(image_path))
#   print(caption_image(image_path))
#   print()
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
# img_path = all_image_paths[1]

# image_path = all_image_paths[0]
# label = all_image_labels[1]
# plt.imshow(load_and_preprocess_image(img_path))
# plt.grid(False)
# plt.xlabel(caption_image(img_path))
# plt.title(label_names[label].title())
# plt.show()


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


BATCH_SIZE = 32
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False


def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names), activation = 'softmax')])

logit_batch = model(image_batch).numpy()
print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.summary()

steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
model.fit(ds, epochs=1, steps_per_epoch=3)
# plt.figure(figsize=(8,8))
# for n, image in enumerate(image_ds.take(4)):
#   plt.subplot(2,2,n+1)
#   plt.imshow(image)
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#   plt.xlabel(caption_image(all_image_paths[n]))
# plt.show()

