import tensorflow as tf
import numpy as np
import pathlib
import random
import os

num_epochs = 30
batch_size = 10
learning_rate = 0.001
data_dir = 'C:/Users/Administrator/Desktop/tensorflow2/numreg/data_dir/train'
data_root = pathlib.Path(data_dir)

def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename) # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string) # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [192,192]) / 255.0
    return image_resized, label

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            
solve_cudnn_error()


image_paths = list(data_root.glob('*/*'))
image_paths =[str(path) for path in image_paths]
train_filenames = tf.constant(image_paths)
#random.shuffle(all_image_paths)

# 列出可用的标签
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in image_paths]
train_labels = tf.constant(image_labels)

index = np.random.permutation(train_filenames.shape[0])
train_filenames = train_filenames.numpy()[index]
train_labels = train_labels.numpy()[index]

train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.map(map_func=_decode_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=23)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=(192, 192, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(8, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

model.fit(train_dataset, epochs=num_epochs)
model.save('G:/works/2020/py-tensorflow/numberdata/saved_model/num_model/saved_model.h5')

# 构建测试数据集
# test_cat_filenames = tf.constant([test_cats_dir + filename for filename in os.listdir(test_cats_dir)])
# test_dog_filenames = tf.constant([test_dogs_dir + filename for filename in os.listdir(test_dogs_dir)])
# test_filenames = tf.concat([test_cat_filenames, test_dog_filenames], axis=-1)
# test_labels = tf.concat([
#     tf.zeros(test_cat_filenames.shape, dtype=tf.int32), 
#     tf.ones(test_dog_filenames.shape, dtype=tf.int32)], 
#     axis=-1)

# test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
# test_dataset = test_dataset.map(_decode_and_resize)
test_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
test_dataset = test_dataset.map(_decode_and_resize)
test_dataset = test_dataset.batch(batch_size)

print(model.metrics_names)
print(model.evaluate(test_dataset))