import tensorflow as tf
import numpy as np

def create_padding_mask(sequences):
    sequences = tf.cast(tf.math.equal(sequences, 0), dtype=tf.float32)
    return sequences[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(seq_len):
    return 1- tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

a = tf.convert_to_tensor(np.array([[1,1,0,0]]))
print(a)
b = create_padding_mask(a)
print(b)

