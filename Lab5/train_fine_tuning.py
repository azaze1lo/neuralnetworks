"""This module implements data feeding and training loop to create model
to classify X-Ray chest images as a lab example for BSU students.
"""

__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2020 Alexander Soroka"""


import argparse
import glob
import numpy as np
import tensorflow as tf
import time
from tensorflow.python import keras as keras
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications import EfficientNetB0
from math import exp
from random import choice
from tensorflow.keras.preprocessing import image

# Avoid greedy memory allocation to allow shared GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


LOG_DIR = 'logs'
CHECKPOINTS_DIR = 'checkpoints'
BATCH_SIZE = 16
NUM_CLASSES = 20
RESIZE_TO = 224
TRAIN_SIZE = 12786


def parse_proto_example(proto):
  keys_to_features = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/label': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }
  example = tf.io.parse_single_example(proto, keys_to_features)
  example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
  example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.uint8)
  example['image'] = tf.image.resize(example['image'], tf.constant([225, 225]))
  return example['image'], tf.one_hot(example['image/label'], depth=NUM_CLASSES)

def crop (image, label):
    return tf.image.random_crop(image, [224,224,3]), label

def contrast(image, label):
    return tf.image.adjust_contrast(image,1.7), label

def brightness (image,label):
    return tf.image.adjust_brightness(image,0.1), label
    
def rotate (image,label):
    return tf.image.rot90(image, choice(range(5))), label

def unfreeze_model(model):
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

def create_dataset(filenames, batch_size):
  """Create dataset from tfrecords file
  :tfrecords_files: Mask to collect tfrecords file of dataset
  :returns: tf.data.Dataset
  """
  return tf.data.TFRecordDataset(filenames)\
    .map(parse_proto_example, num_parallel_calls=tf.data.AUTOTUNE)\
    .cache()\
    .map(contrast)\
    .map(brightness)\
    .map(crop)\
    .map(rotate)\
    .batch(batch_size)\
    .prefetch(tf.data.AUTOTUNE)

def exp_decay(epoch):
    initial_lr = 0.01
    k = 0.25
    lr = initial_lr *exp(-k*epoch)
    return lr

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO,3))
  x = EfficientNetB0(include_top=False, input_tensor=inputs, pooling='avg', weights='imagenet') 
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x.output)
  return tf.keras.Model(inputs=inputs, outputs=outputs)

def main():
  args = argparse.ArgumentParser()
  args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files, use single quote to escape *')
  args = args.parse_args()

  dataset = create_dataset(glob.glob(args.train), BATCH_SIZE)

  train_size = int(TRAIN_SIZE * 0.7 / BATCH_SIZE)
  train_dataset = dataset.take(train_size)
  validation_dataset = dataset.skip(train_size)
  
  model = build_model()
  unfreeze_model(model)

  model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
  )

  log_dir='{}/fine_tuned_model'.format(LOG_DIR)
  checkpoint_path = '{}/wildlife.ckpt'.format(CHECKPOINTS_DIR)
  
  model.load_weights(checkpoint_path)
  
  model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir),
      LearningRateScheduler(exp_decay)
    ]
  )


if __name__ == '__main__':
    main()
