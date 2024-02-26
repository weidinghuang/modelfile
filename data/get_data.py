import tensorflow_datasets as tfds
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']
for pt, en in train_examples.take(1):
  print("Portuguese: ", pt.numpy().decode('utf-8'))
  print("English:   ", en.numpy().decode('utf-8'))
