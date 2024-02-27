import os, sys
from models.transformer import Transformer
from dataset.DataSets import DataSet, Preprocess
import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
def transformer():
    input = tf.keras.Input(shape=(None, ))
    target = tf.keras.Input(shape=(None, ))
    transformer_output = Transformer(8, 30799, 4235)([input, target])
    # final_output = tf.keras.layers.Dense(4235)(transformer_output)
    return tf.keras.Model(inputs=[input, target], outputs=transformer_output)


def masked_loss(label, pred):
    mask = label != 0
    # if tf.executing_eagerly():
    #     print("custom_loss - str(label): \n", str(label))
    #     print("custom_loss - str(pred): \n", str(pred), '\n' * 2)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label,pred)
    # if tf.executing_eagerly():
    #     print("custom_loss00000 - str(loss): \n", str(loss))


    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    # if tf.executing_eagerly():
    #     print("custom_loss - str(loss): \n", str(loss))
    #     print("custom_loss - str(mask): \n", str(mask), '\n' * 2)

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=100):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class evalValidation(tf.keras.callbacks.Callback):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def on_train_begin(self, epochs=None):
        pass





learning_rate = CustomSchedule(768)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-5)

# x = np.array([[101, 10, 11, 12, 13, 14, 15, 102] for _ in range(100)])
# y = np.array([[101, 90, 91, 92, 102, 0, 0, 0] for _ in range(100)])
# train_batches = DataSet(x, y, 2)
# val_batches = DataSet(x, y, 2)
p = Preprocess("data/train_data/translate_data.xlsx")
input = [[source, target] for source, target in zip(p.source[:50], p.target[:50])]
train_batches = DataSet(input, 2)
val_batches = DataSet(input,2)

t = transformer()
t.summary()

t.load_weights("test.hdf5")
# t.add_loss(masked_loss(t.inputs, t.outputs))
#
# t.compile(
#     loss=masked_loss,
#     optimizer=optimizer,
#     metrics=[masked_accuracy],
#     run_eagerly=True
# )
#
# # t.summary()
#
# t.fit(train_batches,
#                 epochs=5,
#                 validation_data=val_batches,
#                 callbacks = [tf.keras.callbacks.ModelCheckpoint("test.hdf5", monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True, verbose=1)])


a = t.predict([np.array([[101, 10, 11, 12, 13, 14, 15, 102]]), np.array([[4227, 0, 0, 0, 0, 0, 0, 0]])])
a = np.array(a)
print(a)
