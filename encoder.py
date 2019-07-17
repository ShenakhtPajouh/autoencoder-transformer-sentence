import tensorflow as tf
import numpy as np
import gpt2 as gpt2

class AttentionAggregation(tf.keras.layers.Layer):

    def __init__(self, num_attention_heads=1, size_per_head=512,
                 query_num=1,
                 initializer_range=0.02,
                 key_act=None,
                 trainable=True,
                 name=None):
        super().__init__(name=name, trainable=trainable)
        # `key_layer` = [B*T, N*H]
        self.key_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=tf.random_normal_initializer(stddev=initializer_range)
        )
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.query_num = query_num
        self.query = None

    def build(self, input_shape):
        self.query = self.add_weight(name="query",
                                     shape=(self.num_attention_heads, self.query_num, self.size_per_head),
                                     initializer=tf.random_normal_initializer(stddev=0.02))
        super().build(input_shape)

    def reshape(self, x, use_2d=False, shape=None):
        if use_2d:
            batch_size, seq_length = shape[0], shape[1]
        else:
            _shape = gpt2.get_tensor_shape(x)
            batch_size, seq_length = _shape[0], _shape[1]
        x = tf.reshape(x, [batch_size, seq_length, self.num_attention_heads, self.size_per_head])
        x = tf.transpose(x, [0, 2, 1, 3])
        return x

    def final_shape(self, x, use_2d=False):
        shape = gpt2.get_tensor_shape(x)
        batch_size, seq_length = shape[0], shape[2]
        x = tf.transpose(x, [0, 2, 1, 3])
        if use_2d:
            x = tf.reshape(x, [batch_size * seq_length, self.num_attention_heads * self.size_per_head])
        else:
            x = tf.reshape(x, [batch_size, seq_length, self.num_attention_heads * self.size_per_head])
        return x

    def get_mask(self, mask):
        if mask is None:
            return mask
        shape = gpt2.get_tensor_shape(mask)
        batch_size, seq_length = shape[0], shape[1]
        mask = tf.reshape(mask, [batch_size, 1, 1, seq_length])
        return mask

    def attend(self, query, key, value, mask=None, dropout=None):
        dim = tf.cast(self.size_per_head, query.dtype)
        _sqrt = tf.math.sqrt(dim)
        _sqrt = tf.cast(_sqrt, query.dtype)
        coefficients = tf.matmul(query, key, transpose_b=True) / _sqrt
        if mask is not None:
            mask = tf.cast(mask, coefficients.dtype)
            coefficients = coefficients * mask - (1 - mask) * 1e5
        coefficients = tf.math.softmax(coefficients, -1)
        coefficients = gpt2.dropout_fn(coefficients, dropout)
        results = tf.matmul(coefficients, value)
        return results

    def call(self, inputs, mask=None, attention_dropout=None, use_2d=False, shape=None):
        if use_2d:
            batch_size = shape[0]
        else:
            batch_size = gpt2.get_tensor_shape(inputs)[0]
        query = tf.tile(tf.expand_dims(self.query, 0), [batch_size, 1, 1, 1])
        key = self.key_layer(inputs)
        value = inputs
        key = self.reshape(key, use_2d, shape)
        value = self.reshape(value, use_2d, shape)
        mask = self.get_mask(mask)
        result = self.attend(query, key, value, mask, attention_dropout)
        result = self.final_shape(result, use_2d)
        return result

    def __call__(self, inputs, mask=None, attention_dropout=None, use_2d=False, shape=None):
        return super().__call__(
            inputs=inputs,
            mask=mask,
            attention_dropout=attention_dropout,
            use_2d=use_2d,
            shape=shape
        )


class Encoder(tf.keras.Model):

    def __init__(self, config, name=None, trainable=True):
        super().__init__(name=name)
        self.trainable = trainable
        self.gpt2 = gpt2.GPT2(config, name="gpt2")
        self.aggregation = AttentionAggregation(
            num_attention_heads=config["n_head"],
            size_per_head=config["n_embd"] // config["n_head"],
            query_num=config["q_num"],
            name="aggregation"
        )


    def call(self, inputs, length=None, dropout=None, attention_dropout=None, use_2d=False):
        shape = gpt2.get_tensor_shape(inputs)
        output = self.gpt2(
            inputs=inputs,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_logits=False,
            use_2d=use_2d
        )
        if length is None:
            mask = None
        else:
            mask = tf.sequence_mask(length, shape[1])
        result = self.aggregation(
            inputs=output,
            mask=mask,
            attention_dropout=attention_dropout,
            use_2d=use_2d,
            shape=shape
        )
        return result


    def __call__(self, inputs, length=None, dropout=None,
                 attention_dropout=None, use_2d=False):
        """
        inputs: an integer tensor of shape [batch_size, seq_length]
        length: a tensor of shape [batch_size]. it is necessary if encode=True
        use_2d: for tpu performances: use 2D tensors for operations and return the output in 2D shape: [batch_size * seq_length, -1]
        """
        return super().__call__(
            inputs=inputs,
            length=length,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_2d=use_2d
        )


