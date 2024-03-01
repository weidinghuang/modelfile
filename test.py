import tensorflow as tf
import numpy as np

a = [[1, 1], [2, 3], [-2*32+1, -2*32+1]]
a = tf.convert_to_tensor(a, dtype='float32')
b = tf.nn.softmax(a, axis=-2)
print(b)