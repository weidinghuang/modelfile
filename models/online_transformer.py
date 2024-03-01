import tensorflow as tf
import numpy as np
import os
### CONFIG ####
BS = 4
#################
class Transformer(tf.keras.Model):
    def __init__(self,
                 inputs_vocab_size,
                 target_vocab_size,
                 encoder_count,
                 decoder_count,
                 attention_head_count,
                 d_model,
                 d_point_wise_ff,
                 dropout_prob):
        """
        :param input_vocab_size
        :param output_vocab_size
        :param encoder_count how many blocks a encoder stack has
        :param decoder_count how many blocks a decode stack has
        :param attention_head_count how many heads a multihead attention has
        :param d_model model dimension
        :param feedforward layer number
        :param dropout rate
        """
        self.inputs_vocab_size = inputs_vocab_size
        self.target_vocab_size = target_vocab_size
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        super(Transformer, self).__init__()

        self.encoder_embedding_layer = Embeddinglayer(inputs_vocab_size, d_model)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)
        self.decoder_embedding_layer = Embeddinglayer(inputs_vocab_size, d_model)
        self.decoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)

        self.encoder_layers = [
            ## concatenate all heads
            EncoderLayer(
                attention_head_count,
                d_model,
                d_point_wise_ff,
                dropout_prob
            ) for _ in range(encoder_count)
        ]
        self.decoder_layers = [
            ## concatenate all head
            DecoderLayer(
                attention_head_count,
                d_model,
                d_point_wise_ff,
                dropout_prob
            ) for _ in range(decoder_count)
        ]

        self.linear = tf.keras.layers.Dense(target_vocab_size)

    def call(self,
             inputs,
             training):
        inputs, target = inputs
        input_padding_mask, look_ahead_mask, target_padding_mask = Mask.create_masks(inputs, target)

        encoder_tensor = self.encoder_embedding_layer(inputs)
        encoder_tensor = self.encoder_embedding_dropout(encoder_tensor)

        for i in range(self.encoder_count):
            encoder_tensor, _ = self.encoder_layers[i](encoder_tensor, inputs_padding_mask, training=training)
        target = self.decoder_embedding_layer(target)
        decoder_tensor = self.decoder_embedding_dropout(target, training=training)
        for i in range(self.decoder_count):
            decoder_tensor, _, _ = self.decoder_layers[i](
                decoder_tensor,
                encoder_tensor,
                look_ahead_mask,
                target_padding_mask,
                training=training
            )
        return self.linear(decoder_tensor)


class Embeddinglayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        super(Embeddinglayer, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, sequence):
        max_sequence_len = sequence.shape[1]
        # output normalized???
        # [B, T, C]
        output = self.embedding(sequence) * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        output = output + self.positional_encoding(max_sequence_len)

    def positional_encoding(self, max_len):

        pos = np.expand_dims(np.arange(0, max_len), axis=1) # [T, 1]
        index = np.expand_dims(np.arange(0, self.d_model), axis=0) # [1, C]
        pe = self.angle(pos, index)


    def angle(self, pos, index):
        return pos / np.power(10000, (index - index % 2) / np.float32(self.d_model))


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        """
        :param attention_head_count:  num of attention head
        :param d_model: model depth
        :param d_point_wise_ff:  pointwise feed forward network
        :param dropout_prob: dropout probability
        """
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(attention_head_count, d_model)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff,
            d_model
        )
        self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        output, attention  = self.multi_head_attention(inputs, inputs, inputs, mask)
        output = self.dropout_1(output, training=training)
        output = self.layer_norm_1(tf.add(inputs, output))
        output_tmp = output

        output = self.position_wise_feed_forward_layer(output)
        output = self.dropout_2(output, training=training)
        output = self.layer_norm_2(tf.add(output_tmp, output))

        return output, attention

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff,
        self.dropout_prob = dropout_prob

        self.masked_multi_head_attention = MultiHeadAttention(attention_head_count, d_model)
        self.dropout_prob_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.encoder_decoder_attention = MultiHeadAttention(attention_head_count, d_model)
        self.dropout_prob_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff, d_model
        )
        self.dropout_prob_3 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        super(DecoderLayer, self).__init__()

    def call(self, decoder_inputs, encoder_output, look_ahead_mask, padding_mask, training):
        output, attention_1 = self.masked_multi_head_attention(
            decoder_inputs,
            decoder_inputs,
            decoder_inputs,
            look_ahead_mask
        )
        output = self.dropout_prob_1(output, training=training)
        query = self.layer_norm_1(tf.add(decoder_inputs, output))

        output, attention_2 = self.encoder_decoder_attention(
            query,
            encoder_output,
            encoder_output,
            padding_mask
        )
        output = self.dropout_prob_2(output, training=training)
        encoder_decoder_attention_output = self.layer_norm_2(tf.add(output, query))

        output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        output = self.dropout_prob_3(output, training=training)
        output = self.layer_norm_3(tf.add(encoder_decoder_attention_output, output))

        return output, attention_1, attention_2


class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model):
        self.w_1 = tf.keras.layers.Dense(d_point_wise_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)
        super(PositionWiseFeedForwardLayer, self).__init__()


    def call(self, inputs):
        inputs = self.w_1(inputs)
        inputs = tf.nn.relu(inputs)
        return self.w_2(inputs)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model):
        """
        :param attention_head_count:
        :param d_model:
        """
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        if d_model % attention_head_count != 0:
            raise ValueError(
                "d_model({}) % attention_head_count({}) is not zero. d_model must be a multiple of attention_head_count".format(
                    d_model, attention_head_count
                )
            )
        super(MultiHeadAttention, self).__init__()

        self.d_h = d_model // attention_head_count # depth of each head
        self.w_query = tf.keras.layers.Dense(d_model)
        self.w_key = tf.keras.layers.Dense(d_model)
        self.w_value = tf.keras.layers.Dense(d_model)

        self.scaled_dot_product = ScaledDotProductAttention(self.d_h)

        self.ff = tf.keras.layers.Dense(d_model)

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0] # [B, T, C]

        query = self.w_query(query)  # [B, T, d_model]
        key = self.w_key(key)
        value = self.w_value(value)

        query = self.split_head(query, batch_size) # [B, T, attentino_head, head_depth]
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        output, attention = self.scaled_dot_product(query, key, value, mask)
        output = self.concat_head(output, batch_size)

        return self.ff(output), attention

    def split_head(self, tensor, batch_size):

        return tf.transpose(
            tf.reshape(
                tensor,
                (batch_size, -1, self.attention_head_count, self.d_h)
            )
        ) [0, 2, 1, 3]  # [B, H, Seq_len, d_h]

    def concat_head(self, tensor, batch_size):
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.attention_head_count*self.d_h)
        )

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_h):
        """

        :param d_h:  depth of each node
        """

        self.d_h = d_h
        super(ScaledDotProductAttention, self).__init__()

    def call(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True) # query [B
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k/scale
        if mask is not None:
            scaled_attention_score += (mask * -1e9)

        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)

        return tf.matmul(attention_weight, value), attention_weight

class Mask:
    @classmethod
    def create_padding_mask(cls, sequences):
        sequences = tf.cast(tf.math.equal(sequences, 0), dtype=tf.float32)
        return sequences[:, tf.newaxis, tf.newaxis, :]

    @classmethod
    def create_look_ahead_mask(cls, seq_len):
        return 1- tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    @classmethod
    def create_masks(cls, inputs, target):
        encoder_padding_mask = Mask.create_padding_mask(inputs)
        decoder_padding_mask = Mask.create_padding_mask(inputs)
        look_ahead_mask = tf.maximum(Mask.create_look_ahead_mask(tf.shape(target)[1]),
                                     Mask.create_padding_mask(target))
        return encoder_padding_mask,look_ahead_mask, decoder_padding_mask


if __name__ == "__main__":
    inputs_vocab_size = 10000
    target_vocab_size = 500
    encoder_count = 6
    decoder_count = 6
    attention_head_count = 8
    d_model = 512
    d_point_wise_ff = 2048
    dropout_prob = 0.1

    transformer = Transformer(
        inputs_vocab_size=inputs_vocab_size,
        target_vocab_size=target_vocab_size,
        encoder_count=encoder_count,
        decoder_count=decoder_count,
        attention_head_count=attention_head_count,
        d_model=d_model,
        d_point_wise_ff=d_point_wise_ff,
        dropout_prob=dropout_prob
    )
    transformer.summary()
