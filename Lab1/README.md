# отчет

## график 1
![Figure 1](./epoch_categorical_accuracy.svg)

## график 2
![Figure 2](./epoch_loss.svg)

## архитектура
``` inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x1)
  x3 = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x2)
  x4 = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x3)
  x5 = tf.keras.layers.MaxPool2D()(x4)
  x6 = tf.keras.layers.Flatten()(x5)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x6)
