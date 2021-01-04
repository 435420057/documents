from cv2 import dnn
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import pathlib
import random


print(tf.__version__)
print(cv.__version__)

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


model = tf.keras.models.load_model("G:/works/2020/py-tensorflow/numberdata/saved_model/num_model/1")
test_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
test_dataset = test_dataset.map(_decode_and_resize)
test_dataset = test_dataset.batch(batch_size)



# sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
# num_batches = 1
# for batch_index in range(num_batches):
#     start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
#     y_pred = model(test_dataset)
#     sparse_categorical_accuracy.update_state(y_true=test_dataset, y_pred=y_pred)
# print("test accuracy: %f" % sparse_categorical_accuracy.result())

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(x=(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)))

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="G:/works/2020/py-tensorflow/numberdata",
                  name="saved_model.pbtxt",
                  as_text=False)