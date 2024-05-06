import tensorflow as tf
import numpy as np
tf.config.run_functions_eagerly(True)
def get_positional_embedding(max_input_length, hidden_size):
    positions = tf.expand_dims(tf.range(max_input_length, name="pos_embedding_x"), 1) #[seq_len, 1]
    angles = tf.expand_dims(1/(1000**(2*tf.range(hidden_size/2)/hidden_size)), 0) # [1, d_model//2, ]
    positions = tf.cast(positions, dtype="float32")
    angle_rates = positions * angles
    pos_embedding = tf.keras.backend.concatenate([tf.keras.backend.sin(angle_rates), tf.keras.backend.cos(angle_rates)], axis=-1)
    return pos_embedding


def reshape_attention_shape(input, head_size=None, flag=True, hidden_size=768):
    '''
    input_size = [BS, seq_len, hidden_size] -> [BS, head_size, seq_len, hidden_size//head_size]
    flag: to expand dimensions or squeeze dimensions
    '''
    if flag:
        batch_size, seq_length, hidden_size = tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2]
        head_size, hidden_size = 8, 768
        input = tf.reshape(input, [batch_size, seq_length, head_size, hidden_size//head_size])
        input = tf.transpose(input, perm=[0, 2, 1, 3])
    else:
        batch_size, head_size, seq_length, depth = tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]
        head_size, depth = 8, 96
        input = tf.transpose(input, perm=[0, 2, 1, 3])
        input = tf.reshape(input, [batch_size, seq_length, head_size*depth])
    return input
class tokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size) -> None:
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size, mask_zero=True)

    def call(self, inputs):
        return self.embedding(inputs)



class positionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, hidden_size) -> None:
        self.hidden_size = hidden_size
        super(positionalEmbedding, self).__init__()

    # def build(self, input_shape):
    #     self.positional_embedding = get_positional_embedding(input_shape[1], self.hidden_size)
    #     super(positionalEmbedding, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        positions = tf.expand_dims(tf.range(seq_length, name="pos_embedding_x"), 1) #[seq_len, 1]
        angles = tf.expand_dims(1/(1000**(2*tf.range(self.hidden_size/2)/self.hidden_size)), 0) # [1, d_model//2, ]
        positions = tf.cast(positions, dtype="float32")
        angle_rates = positions * angles
        pos_embedding = tf.keras.backend.concatenate([tf.keras.backend.sin(angle_rates), tf.keras.backend.cos(angle_rates)], axis=-1)
        return tf.tile(tf.expand_dims(pos_embedding, 0), [batch_size, 1, 1])

class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, head_size) -> None:
        self.hidden_size = hidden_size
        self.head_size = head_size
        super(DotProductAttention, self).__init__()

    def call(self, inputs, mask=None):
        query, key, value = inputs
        scores = tf.linalg.matmul(query, key, transpose_b=True) # [BS, seq_length, hidden_size] * [BS, seq_length, hidden_size] -> [BS, seq_length, seq_length]
        hidden_size = tf.cast(self.hidden_size, dtype='float32')
        scores /= tf.math.sqrt(hidden_size)
        # if mask is not None: # [BS, seq_length]
        #     mask = tf.cast(mask, dtype="float32")
        #     mask = mask[:, tf.newaxis, tf.newaxis, :]
        #     scores += -1e9 * (1-mask)
        scores += -1e9*mask
        weights = tf.keras.backend.softmax(scores)
        return tf.linalg.matmul(weights, value)

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, head_size) -> None:
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.wq = tf.keras.layers.Dense(hidden_size)
        self.wk = tf.keras.layers.Dense(hidden_size)
        self.wv = tf.keras.layers.Dense(hidden_size)

    def call(self, inputs, mask=None):
        x, context_output, context_output = inputs # x is after self attention with time series masking
        # q, k, v
        #  look-ahead mask for q , and padding mask for k and v
        query, key, value = self.split_head(self.wq(x)), self.split_head(self.wk(context_output)), self.split_head(self.wv(context_output))
        # if mask is not None:
        #     mask = mask[1]
        #     mask = tf.cast(mask, dtype="float32")
        #     mask = mask[:, tf.newaxis, tf.newaxis, :]
        attention_result= self.dot_product(query, key, value, mask)
        attention_result = self.concat_head(attention_result)
        return attention_result

    def split_head(self, input):
        # [BS, seq_length, hidden_size)
        BS = tf.shape(input)[0]
        new_input = tf.reshape(input, [BS, -1, self.head_size, self.hidden_size // self.head_size])
        return tf.transpose(new_input, [0, 2, 1, 3])

    def concat_head(self, input):
        BS = tf.shape(input)[0]
        new_input = tf.transpose(input, [0, 2, 1, 3])
        return tf.reshape(new_input, [BS, -1, self.hidden_size])

    def dot_product(self, query, key, value, mask):
        scores = tf.linalg.matmul(query, key, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.hidden_size, dtype='float32'))
        scores += -1e9 * mask
        weights = tf.keras.backend.softmax(scores)
        return tf.linalg.matmul(weights, value)


class decoderSelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, head_size) -> None:
        super(decoderSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.wq = tf.keras.layers.Dense(hidden_size)
        self.wk = tf.keras.layers.Dense(hidden_size)
        self.wv = tf.keras.layers.Dense(hidden_size)


    def call(self, inputs, mask=None):
        x, y, z = inputs
        x_seq_length = tf.shape(x)[1] #x.shape.as_list[0] # [BS, seq_length, hidden_size]
        # time_mask = tf.ones(shape=(x_seq_length, x_seq_length))
        # time_mask = 1 - tf.linalg.band_part(time_mask, -1, 0)
        query, key, value = self.split_head(x), self.split_head(y), self.split_head(z)
        # if mask is not None:
        #     mask = mask[1]
        #     mask = 1-mask
        #     mask = mask[:, tf.newaxis, tf.newaxis, :]
        #     new_mask = tf.maximum(time_mask, mask)
        #     # new_mask = new_mask[:, tf.newaxis, tf.newaxis, :]
        #     mask = new_mask
        mask = -1e9 * mask
        attention_result = self.dot_product(query, value, key, mask)
        attention_result = self.concat_head(attention_result)
        return attention_result

    def split_head(self, input):
        # [BS, seq_length, hidden_size)
        BS = tf.shape(input)[0]
        new_input = tf.reshape(input, [BS, -1, self.head_size, self.hidden_size//self.head_size])
        return tf.transpose(new_input, [0, 2, 1, 3])

    def concat_head(self, input):
        BS = tf.shape(input)[0]
        new_input = tf.transpose(input, [0, 2, 1, 3])
        new_input = tf.reshape(new_input, [BS, -1, self.hidden_size])
        return new_input

    def dot_product(self, query, key, value, mask):
        scores = tf.linalg.matmul(query, key, transpose_b=True) /tf.math.sqrt(tf.cast(self.hidden_size, dtype='float32'))
        scores += -1e9 * mask
        weights = tf.keras.backend.softmax(scores)
        return tf.linalg.matmul(weights, value)


class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, head_size) -> None:
        self.dot_product_attention = DotProductAttention(hidden_size, head_size)
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.wq = tf.keras.layers.Dense(hidden_size)
        self.wk = tf.keras.layers.Dense(hidden_size)
        self.wv = tf.keras.layers.Dense(hidden_size)
        # self.wo = tf.keras.layers.Dense(hidden_size)
        super(Attention, self).__init__()

    def call(self, inputs, mask=None, **kwarg):
        q, k, v = inputs
        query_reshaped = self.split_head(self.wq(q))
        key_reshaped = self.split_head(self.wk(k))
        value_reshaped = self.split_head(self.wv(v))
        attention_embedding = self.dot_product_attention([query_reshaped, key_reshaped, value_reshaped], mask=mask)

        output = self.concat_head(attention_embedding)
        return output

    def split_head(self, input):
        # [BS, seq_length, hidden_size)
        BS = tf.shape(input)[0]
        new_input = tf.reshape(input, [BS, -1, self.head_size, self.hidden_size//self.head_size])
        return tf.transpose(new_input, [0, 2, 1, 3])

    def concat_head(self, input):
        BS = tf.shape(input)[0]
        new_input = tf.transpose(input, [0, 2, 1, 3])
        return tf.reshape(new_input, [BS, -1, self.hidden_size])

class inputEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_length, vocab_size, hidden_size) -> None:
        super(inputEmbedding, self).__init__()
        self.input_length = input_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, hidden_size, mask_zero=True)
        self.positional_embedding = positionalEmbedding(hidden_size)
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        return self.add([self.token_embedding(inputs), self.positional_embedding(inputs)])

class encoderLayer(tf.keras.layers.Layer):
    def __init__(self, input_length, num_blocks, hidden_size, head_size, dropout_rate, vocab_size, name=None):
        super().__init__(name=name)
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.encoder_attention_list = Attention(hidden_size, head_size)
        self.add_1 = tf.keras.layers.Add()
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.dense_up1 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense_down1 = tf.keras.layers.Dense(hidden_size)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_up = tf.keras.layers.Dense(2048, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_down = tf.keras.layers.Dense(hidden_size)
        self.add_2 = tf.keras.layers.Add()
        self.norm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        input_embedding = inputs # input size = [BS, SEQ_LENGTH, HIDDEN_SIZE]
        output = self.encoder_attention_list([input_embedding, input_embedding, input_embedding], mask=mask)
        output = self.dropout1(output)
        output = self.add_1([input_embedding, output])
        intermediate_output = self.norm_1(output)
        output = self.add_2([output, intermediate_output])
        input_embedding = self.norm_2(output)
        return input_embedding

class encoder(tf.keras.layers.Layer):
    def __init__(self, input_length, num_blocks, hidden_size, head_size, dropout_rate, vocab_size) -> None:
        super().__init__()
        # cannot determine input length
        self.num_blocks = num_blocks
        self.encoder_attention_list = [encoderLayer(input_length,
                                                   num_blocks,
                                                   hidden_size,
                                                   head_size,
                                                   dropout_rate,
                                                   vocab_size, name=str(i)) for i in range(num_blocks)]
        self.input_embedding = inputEmbedding(input_length, vocab_size, hidden_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)


    def call(self, inputs, mask=None):
        # input embeddings added
        input_embedding = self.input_embedding(inputs)
        input_embedding = self.dropout(input_embedding)
        for i in range(self.num_blocks):
            input_embedding = self.encoder_attention_list[i](input_embedding, mask=mask)
        return input_embedding

class decoderLayer(tf.keras.layers.Layer):
    def __init__(self, input_length, num_blocks, hidden_size, head_size, dropout_rate, vocab_size, name=None) -> None:
        super().__init__(name=name)
        self.input_length = input_length
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.decoder_self_attention_list = decoderSelfAttention(hidden_size, head_size)
        self.decoder_cross_attention_list = CrossAttention(hidden_size, head_size)
        self.add_1 = tf.keras.layers.Add()
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.add_2 = tf.keras.layers.Add()
        self.norm_2 = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(hidden_size)
        self.add_3 = tf.keras.layers.Add()
        self.norm_3 = tf.keras.layers.LayerNormalization()

        self.dense_up = tf.keras.layers.Dense(2048, activation='relu')
        self.dense_down = tf.keras.layers.Dense(hidden_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.dense_up1 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense_down1 = tf.keras.layers.Dense(hidden_size)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_up2 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense_down2 = tf.keras.layers.Dense(hidden_size)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, mask=[]):
        decoder_input_embedding, context = inputs
        decoder_input = self.decoder_self_attention_list([decoder_input_embedding, decoder_input_embedding, decoder_input_embedding], mask=mask[0])
        decoder_input = self.dense_up(decoder_input)
        decoder_input = self.dense_down(decoder_input)
        decoder_input = self.dropout(decoder_input)
        decoder_input = self.add_1([decoder_input_embedding, decoder_input])
        decoder_input = self.norm_1(decoder_input)
        output = self.decoder_cross_attention_list([decoder_input, context, context], mask=mask[1])
        output = self.dense_up1(output)
        output = self.dense_down1(output)
        output = self.dropout1(output)
        output = self.add_2([decoder_input, output])
        cross_output = self.norm_2(output)
        final_output = self.dense_up2(cross_output)
        final_output = self.dense_down2(final_output)
        final_output = self.dropout2(final_output)
        final_output = self.add_3([cross_output, final_output])
        decoder_input_embedding = self.norm_3(final_output)

        return decoder_input_embedding


class decoder(tf.keras.layers.Layer):
    def __init__(self, input_length, num_blocks, hidden_size, head_size, dropout_rate, vocab_size) -> None:
        super().__init__()
        self.decoder_layer_list = [decoderLayer(input_length,
                                                num_blocks,
                                                hidden_size,
                                                head_size,
                                                dropout_rate,
                                                vocab_size,
                                                name=str(i))
                                   for i in range(num_blocks)
                                   ]
        self.num_blocks = num_blocks
        self.decoder_input_embedding = inputEmbedding(input_length, vocab_size, hidden_size)


    def call(self, inputs, mask=None):
        context, x = inputs
        decoder_input_embedding = self.decoder_input_embedding(x)
        for i in range(self.num_blocks):
            decoder_input_embedding=self.decoder_layer_list[i]([decoder_input_embedding, context], mask=mask)
        return decoder_input_embedding


class Transformer(tf.keras.layers.Layer):
    def __init__(self, input_length, input_vocab_size, target_vocab_size, hidden_size=768, head_size=8, num_blocks=1, dropout_rate=0.2, **kwargs):
        super(Transformer, self).__init__(**kwargs) 
        self.input_length = input_length
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.hidden_size = head_size
        self.head_size = head_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.encoder = encoder(input_length, num_blocks, hidden_size, head_size, dropout_rate, input_vocab_size)
        self.decoder = decoder(input_length, num_blocks, hidden_size, head_size, dropout_rate, target_vocab_size)

    def call(self, inputs):
        encoder_input, decoder_input = inputs
        encoder_padding_mask, look_ahead_mask, cross_attention_mask = self.get_masks(encoder_input, decoder_input)
        encoder_output = self.encoder(encoder_input, mask=encoder_padding_mask)
        decoder_output = self.decoder([encoder_output, decoder_input], mask=[look_ahead_mask, cross_attention_mask])
        # final_output = tf.keras.layers.Dense(4235, activation='softmax')(decoder_output)
        return decoder_output
    
    def get_config(self):

        config = super().get_config().copy()
        # config.update({
        #     'vocab_size': self.vocab_size,
        #     'num_layers': self.num_layers,
        #     'units': self.units,
        #     'd_model': self.d_model,
        #     'num_heads': self.num_heads,
        #     'dropout': self.dropout,
        # })
        return config
    
    def get_masks(self, encoder_input, decoder_input):
        # encoder input dimension [BS, SEQ_LENGTH]
        encoder_padding_mask, look_ahead_mask, cross_attention_mask = self.create_masks(encoder_input, decoder_input)

        return encoder_padding_mask, look_ahead_mask, cross_attention_mask
    
    def create_padding_mask(self, sequences):
        sequences = tf.cast(tf.math.equal(sequences, 0), dtype=tf.float32)
        return sequences[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq_len):
        return 1- tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    def create_masks(self, inputs, target):
        encoder_padding_mask = self.create_padding_mask(inputs)
        decoder_padding_mask = self.create_padding_mask(inputs)
        look_ahead_mask = tf.maximum(self.create_look_ahead_mask(tf.shape(target)[1]),
                                     self.create_padding_mask(target))
        return encoder_padding_mask,look_ahead_mask, decoder_padding_mask

def build_bert():
    input_token = tf.keras.Input(shape=(None, ), name="input_token")
    input_embedding = tf.keras.layers.Embedding(150, 2000)(input_token)
    return tf.keras.Model(inputs=input_token, outputs=input_embedding)


if __name__ == "__main__":
    import sys
    sys.path.insert(0,"../")
    from dataset.DataSets import bert_dataset, new_dataset
    from dataset.trans_dataset import DataSet_trilabels
    import tensorflow as tf
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.callbacks import ModelCheckpoint
    def transformer():
        input = tf.keras.Input(shape=(None, ))
        target = tf.keras.Input(shape=(None, ))
        transformer_output = Transformer(50, 163, 88)([input, target])
        # output = Lambda(lambda x: x[:, 0, :])(transformer_output)
        output = tf.keras.layers.Dense(88, activation='softmax')(transformer_output)
        # output = tf.argmax(output, axis=-1)
        # output = Lambda(lambda x: x[0, :, :])(output)

        return tf.keras.Model(inputs=[input, target], outputs=output)
    model = transformer()
    # model.summary()
    # model.compile(optimizer=tf.keras.optimizers.SGD(), loss='categorical_crossentropy', metrics='acc')
    # dataset = new_dataset("../data/raw_data/formatted_data.txt")
    # dataset.load_vocab("../dataset/source_vocab.txt", "../dataset/target_vocab.txt")
    # train_gen = tf.data.Dataset.from_generator(dataset.train_gen,
    #                                            output_signature=(
    #      (tf.TensorSpec(shape=(None, None), dtype=tf.int32),tf.TensorSpec(shape=(None, None), dtype=tf.int32)),
    #      tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)))
    # model.fit_generator(train_gen, epochs=10, steps_per_epoch=1000//8,
    #                     callbacks=[
    #                             ModelCheckpoint('test.hdf5',
    #                                             monitor='acc',
    #                                             mode='max',
    #                                             save_best_only=True,
    #                                             save_weights_only=True,
    #                                             verbose=1)])
    model.load_weights("test.hdf5")
    input = np.array([[69, 156, 15, 82, 79, 160, 162, 51, 79, 162, 142, 79, 132, 143, 79, 122, 11, 51, 82, 79, 127, 162, 160, 120, 60, 162, 157, 132, 138, 82, 138, 144, 138, 143, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    target = np.array([[86, 12, 85, 85, 37]])
    b = 0
    while b != 87:
        a = model.predict((input, target))
        print(a)
        b = np.argmax(a, -1)[:, -1]
        print(b)
        break
        # target += b
    #
    # model = tf.keras.models.Sequential(model1.run_eagerly = True)
    # model.add(tf.keras.layers.Embedding(150, 2000))
    # model.add(tf.keras.layers.Lambda(lambda x: x[:, 0, :]))
    # model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # model.summary()
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')
    # model.fit(bert_dataset.data_generator(), epochs=1, steps_per_epoch=100, callbacks=[tf.keras.callbacks.ModelCheckpoint('bert_test.hdf5', monitor='acc',save_best_only=True)])
    # print(model.predict([[101, 1, 2, 3, 4, 5, 6, 102]]))
    # print(model.predict([[101, 6, 5, 4, 3 ,2 ,1,102]]))
    # t = transformer()
    # t.summary()
    # t.compile(optimizer=tf.keras.optimizers.SGD(), loss='categorical_crossentropy', metrics='acc')
    # t.fit(bert_dataset.data_generator(), epochs=5, steps_per_epoch=100, callbacks=[tf.keras.callbacks.ModelCheckpoint('bert_test.hdf5', monitor='acc',save_best_only=True)])
    # # t.load_weights("bert_test.hdf5")
    # print(t.predict([[101, 1, 2, 3, 4, 5, 6, 102]]))

#     train_data = DataSet_trilabels(vocab_file='/repos/model/uncased_L-12_H-768_A-12/vocab.txt', 
#                                 data_type='train', 
#                                 do_lower_case=True,
#                                 label_path= ["data/ibu_hotel/train_labels/first_layer_labels.txt", 
#                                                 "data/ibu_hotel/train_labels/second_layer_labels.txt", 
#                                                 "data/ibu_hotel/train_labels/third_layer_labels.txt"]
#                                 )
#     train_data.read_from_file("data/ibu_hotel/train_data/all_data.txt")
#     valid_data = DataSet_trilabels(vocab_file='/repos/model/uncased_L-12_H-768_A-12/vocab.txt', 
#                                 data_type='valid',
#                                 do_lower_case=True,
#                                 label_path= ["data/ibu_hotel/train_labels/first_layer_labels.txt", 
#                                                 "data/ibu_hotel/train_labels/second_layer_labels.txt", 
#                                                 "data/ibu_hotel/train_labels/third_layer_labels.txt"]
#                                 )
#     valid_data.read_from_file("data/ibu_hotel/train_data/all_data.txt")
#
#
# # from datetime import datetime
# # from tensorflow import keras
# # logdir="./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
#
# # firstLayerOutput = calculateClassWeights("data/ibu_hotel/train_.txt", "data/ibu_hotel/train_labels/first_layer_labels.txt", 0)
# # secondLayerOutput = calculateClassWeights("data/ibu_hotel/test.txt", "data/ibu_hotel/train_labels/second_layer_labels.txt",1)
# # secondLayerOutput = calculateClassWeights("data/ibu_hotel/test.txt", "data/ibu_hotel/train_labels/third_layer_labels.txt",2)
#     model.fit_generator(train_data.data_generator(), 
#                         epochs=1, 
#                         steps_per_epoch=np.ceil(train_data.count/CONFIG.BatchSize), 
#                         validation_data=valid_data.data_generator(), 
#                         validation_steps=np.ceil(valid_data.count/CONFIG.BatchSize),
#                         callbacks=[custom_callback(),
#                                 ModelCheckpoint('test.hdf5', 
#                                                 monitor='val_f1', 
#                                                 mode='max', 
#                                                 save_best_only=True, 
#                                                 save_weights_only=True, 
#                                                 verbose=1)])
