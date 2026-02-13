import tensorflow as tf

from cutting_layer import last_output

# flattening the output of last layer into 1D vector

inputs = tf.keras.layers.Flatten()(last_output)

# Adding 1 hidden layer with 256 neurons and activation function relu

x = tf.keras.layers.Dense(256, activation = 'relu')(inputs)

# Dropout layer with 0.2 value to drop 20% random neurons while training to avoid overfitting

x = tf.keras.layers.Dropout(0.2)(x)

# Adding output layer
x = tf.keras.layers.Dense(1, activation= 'sigmoid')(x)

print("custom layers defined")
