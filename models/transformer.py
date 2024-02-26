import tensorflow as tf
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
        input = tf.reshape(input, [batch_size, seq_length, head_size, hidden_size//head_size])
        input = tf.transpose(input, perm=[0, 2, 1, 3])
    else:
        batch_size, head_size, seq_length, depth = tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]
        input = tf.transpose(input, perm=[0, 2, 1, 3])
        input = tf.reshape(input, [batch_size, seq_length, head_size*depth])
    return input

def encoder_stack(inputs, num_blocks, hidden_size, head_size, dropout_rate):
    for i in range(num_blocks):
        output = Attention(hidden_size, head_size)(inputs)
        attention_output = tf.keras.layers.Add()([inputs[0], output])
        attention_output = tf.keras.layers.LayerNormalization()(attention_output)
        attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
        intermediate_output = tf.keras.layers.Dense(hidden_size)(tf.keras.layers.Dense(3072)(attention_output))
        intermediate_add = tf.keras.layers.Add()([attention_output, intermediate_output])
        intermediate_layernorm = tf.keras.layers.LayerNormalization()(intermediate_add)
        attention_output = tf.keras.layers.Dropout(dropout_rate)(intermediate_layernorm)
        inputs = [attention_output, attention_output, attention_output]

    return inputs[0]

def decoder_stack(encoder_output, inputs, num_blocks, hidden_size, head_size, dropout_rate):
    # decoder self attention
    for i in range(num_blocks):
        output = decoderSelfAttention(hidden_size, head_size)(inputs) # inputs = [[], [], []]
        output = tf.keras.layers.Add(inputs[0], output) # inputs[0] is the query emebedding
        output = tf.keras.layers.LayerNormalization()(output)
        output = tf.keras.layers.Dropout(dropout_rate)(output)
        cross_attention_output = CrossAttention(hidden_size, head_size)([output, encoder_output, encoder_output])
        cross_attention_output = tf.keras.layers.Add(output, cross_attention_output) # inputs[0] is the query emebedding
        cross_attention_output = tf.keras.layers.LayerNormalization()(cross_attention_output)
        cross_attention_output = tf.keras.layers.Dropout(dropout_rate)(cross_attention_output)
        inputs = [cross_attention_output, cross_attention_output, cross_attention_output]
    return inputs[0]
        


class tokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size) -> None:
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size, mask_zero=True)

    def call(self, inputs):
        return self.embedding(inputs)



class positionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_length, hidden_size) -> None:
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        super(positionalEmbedding, self).__init__()

    def build(self, input_shape):
        self.positional_embedding = get_positional_embedding(self.seq_length, self.hidden_size)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.tile(tf.expand_dims(self.positional_embedding, 0), [batch_size, 1, 1])

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
        if mask is not None: # [BS, seq_length]
            mask = mask[0]
            mask = tf.cast(mask, dtype="float32")
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.tile(mask, [1, self.head_size, 1, 1])
            scores += -1e9 * (1-mask)
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

    # def build(self, input_shape):
    #     self.wq = self.add_weight(shape=(self.hidden_size), name="CrossAttention_Query_Kernel")
    #     self.wk = self.add_weight(shape=(self.hidden_size), name="CrossAttention_Key_Kernel")
    #     self.wv = self.add_weight(shape=(self.hidden_size), name="CrossAttention_Value_Kernel")
    #     super(CrossAttention, self).__init__()

    def call(self, inputs, mask=None):
        x, context_output, context_output = inputs # x is after self attention with time series masking
        # create a lower triangluar matrix mask
        query = self.wq(x)
        key, value = self.wk(context_output), self.wv(context_output)
        scores = tf.linalg.matmul(query, key, transpose_b=True)
        if mask is not None:
            mask = mask[0]
            mask = tf.cast(mask, dtype="float32")
            mask = tf.expand_dims(mask, axis=-1)
            scores += -1e9 * (1-mask)
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

    # def build(self, input_shape):
    #     self.wq = self.add_weight(shape=(self.hidden_size), name="CrossAttention_Query_Kernel")
    #     self.wk = self.add_weight(shape=(self.hidden_size), name="CrossAttention_Key_Kernel")
    #     self.wv = self.add_weight(shape=(self.hidden_size), name="CrossAttention_Value_Kernel")
    #     super(decoderSelfAttention, self).__init__()

    def call(self, inputs, mask=None):
        x, _, _ = inputs
        x_seq_length = tf.shape(x)[1] #x.shape.as_list[0] # [BS, seq_length, hidden_size]
        time_mask = tf.ones(shape=(x_seq_length, x_seq_length))
        time_mask = tf.linalg.band_part(time_mask, -1, 0)
        time_mask = tf.expand_dims(time_mask, axis=0) # [1, seq_length, seq_length]
        query = self.wq(x)
        key, value = self.wk(x), self.wv(x)
        hidden_size = tf.cast(self.hidden_size, dtype='float32')
        scores = tf.linalg.matmul(query, key, transpose_b=True)
        scores /= tf.math.sqrt(hidden_size)
        if mask is not None:
            mask = mask[0]
            mask = tf.cast(mask, dtype="float32")
            mask = tf.expand_dims(mask, axis=-1)
            mask = (1-time_mask) * mask
            scores += -1e9 * (1-mask)
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
        query_reshaped = reshape_attention_shape(self.wq(q), self.head_size, True)
        key_reshaped = reshape_attention_shape(self.wk(k), self.head_size, True)
        value_reshaped = reshape_attention_shape(self.wv(v), self.head_size, True)
        attention_embedding = self.dot_product_attention([query_reshaped, key_reshaped, value_reshaped], mask=mask)

        output = reshape_attention_shape(attention_embedding, flag=False)
        return output

def bert(max_input_length, vocab_size, hidden_size=768, head_size=8, num_blocks=12, dropout_rate=0.2):
    input_tokens = tf.keras.Input(shape=(None, ), name="Input_Tokens")
    segment_tokens = tf.keras.Input(shape=(None,), name="Segment_Tokens")
    input_embedding = tf.keras.layers.Embedding(vocab_size, hidden_size, mask_zero=True, name="Input_Embedding")(input_tokens)
    segment_embedding = tf.keras.layers.Embedding(2, hidden_size, name="Segment_Embedding")(segment_tokens)
    positional_embedding = positionalEmbedding(max_input_length, hidden_size)(input_embedding)

    embeddings = tf.keras.layers.Add()([input_embedding, segment_embedding, positional_embedding])
    encoder_output = encoder_stack([embeddings, embeddings, embeddings], num_blocks, hidden_size, head_size, dropout_rate)

    return tf.keras.Model(inputs=[input_tokens, segment_tokens], outputs=[encoder_output])

class inputEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_length, vocab_size, hidden_size) -> None:
        super(inputEmbedding, self).__init__()
        self.input_length = input_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, hidden_size, mask_zero=True)
        self.positional_embedding = positionalEmbedding(input_length, hidden_size)
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        return self.add([self.token_embedding(inputs), self.positional_embedding(inputs)])

class encoder(tf.keras.layers.Layer):
    def __init__(self, input_length, num_blocks, hidden_size, head_size, dropout_rate, vocab_size) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.input_embedding = inputEmbedding(input_length, vocab_size, hidden_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_attention_list = [Attention(hidden_size, head_size) for _ in range(num_blocks)]
        self.add_1 = tf.keras.layers.Add()
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(hidden_size)
        self.add_2 = tf.keras.layers.Add()
        self.norm_2 = tf.keras.layers.LayerNormalization()
        
    def call(self, inputs):
        # input embeddings added
        input_embedding = self.input_embedding(inputs)
        input_embedding = self.dropout(input_embedding)
        for i in range(self.num_blocks):
            output = self.encoder_attention_list[i]([input_embedding, input_embedding, input_embedding])
            output = self.add_1([input_embedding, output])
            output = self.norm_1(output)
            intermediate_output = self.dense(output)
            output = self.add_2([output, intermediate_output])
            input_embedding = self.norm_2(output)

        return input_embedding

class decoder(tf.keras.layers.Layer):
    def __init__(self, input_length, num_blocks, hidden_size, head_size, dropout_rate, vocab_size) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.decoder_input_embedding = inputEmbedding(input_length, vocab_size, hidden_size)
        self.decoder_self_attention_list = [decoderSelfAttention(hidden_size, head_size) for _ in range(num_blocks)]
        self.decoder_cross_attention_list = [CrossAttention(hidden_size, head_size) for _ in range(num_blocks)]
        self.add_1 = tf.keras.layers.Add()
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.add_2 = tf.keras.layers.Add()
        self.norm_2 = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(hidden_size)
        self.add_3 = tf.keras.layers.Add()
        self.norm_3 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        context, x = inputs
        decoder_input_embedding = self.decoder_input_embedding(x)
        for i in range(self.num_blocks):
            decoder_input = self.decoder_self_attention_list[i]([decoder_input_embedding, decoder_input_embedding, decoder_input_embedding], mask=mask)
            decoder_input = self.add_1([decoder_input_embedding, decoder_input])
            decoder_input = self.norm_1(decoder_input)
            output = self.decoder_cross_attention_list[i]([decoder_input, context, context], mask=mask)
            output = self.add_2([decoder_input, output])
            cross_output = self.norm_2(output)
            final_output = self.dense(cross_output)
            final_output = self.add_3([cross_output, final_output])
            decoder_input_embedding = self.norm_3(final_output)

        return decoder_input_embedding


class Transformer(tf.keras.layers.Layer):
    def __init__(self, input_length, input_vocab_size, target_vocab_size, hidden_size=768, head_size=8, num_blocks=12, dropout_rate=0.2) -> None:
        super(Transformer, self).__init__() 
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
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder([encoder_output, decoder_input])
        return decoder_output 



if __name__ == "__main__":
    def transformer():
        input = tf.keras.Input(shape=(None, ))
        target = tf.keras.Input(shape=(None, ))
        transformer_output = Transformer(8, 3000, 2000)([input, target])
        return tf.keras.Model(inputs=[input, target], outputs=transformer_output)
    t = transformer()
    t.summary()

