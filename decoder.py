import tensorflow as tf
import numpy as np
import gpt2


class FeederBlock(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads=1,
                 size_per_head=512,
                 initializer_range=0.02,
                 value_act=None,
                 key_act=None,
                 trainable=True,
                 name=None):
        super().__init__(name=name, trainable=trainable)
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.attention_size = num_attention_heads * size_per_head
        self.layer_norm = gpt2.LayerNormalization(name="layer_norm")
        self.key_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=tf.random_normal_initializer(stddev=initializer_range)
        )
        self.value_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            kernel_initializer=tf.random_normal_initializer(stddev=initializer_range)
        )

    def call(self, inputs, use_2d=False, shape=None):
        if use_2d and shape is None:
            raise ValueError("if use_2d is True, shape must me specified.")
        x = self.layer_norm(inputs)
        key = self.key_layer(x)
        value = self.value_layer(x)
        if not use_2d:
            shape = gpt2.get_tensor_shape(inputs)
        key = tf.reshape(key, [shape[0], shape[1], self.num_attention_heads, self.size_per_head])
        value = tf.reshape(value, [shape[0], shape[1], self.num_attention_heads, self.size_per_head])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])
        result = {"key": key, "value": value}
        return result

    def __call__(self, inputs, use_2d=False, shape=None):
       return super().__call__(
           inputs=inputs,
           use_2d=use_2d,
           shape=shape
       )

class Feeder(tf.keras.Model):
    def __init__(self, config, name=None, trainable=True):
        super().__init__(name=name)
        self.trainable = trainable
        self.blocks = []
        for i in range(config["n_layer"]):
            block = FeederBlock(
                num_attention_heads=config["n_head"],
                size_per_head=config["n_embd"] // config["n_head"],
                name="block_" + str(i)
            )
            self.blocks.append(block)

    def call(self, inputs, use_2d=False, shape=None):
        result = []
        for block in self.blocks:
            res = block(
                inputs=inputs,
                use_2d=use_2d,
                shape=shape
            )
            result.append(res)
        return result

    def __call__(self, inputs, use_2d=False, shape=None):
        return super().__call__(
            inputs=inputs,
            use_2d=use_2d,
            shape=shape
        )



class ModifiedGPT2(tf.keras.Model):

    def __init__(self, config, name=None, trainable=True, dtype=None):
        super().__init__(name=name)
        self.trainable = trainable
        self.embedding = gpt2.Embedding(
            embedding_size=config['n_embd'],
            vocab_size=config['n_vocab'],
            max_position_length=config['n_ctx'],
            name="embedding",
            dtype=dtype
        )
        self.transformer = gpt2.Transformer(config, name="transformer")
        self.start_token = None
        self.end_token = None

    def build(self, input_shape):
        self.start_token = self.add_weight(
            name="start_token",
            shape=(self.embedding.embedding_size, ),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=0.02)
        )
        self.end_token = self.add_weight(
            name="end_token",
            shape=(self.embedding.embedding_size, ),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=0.02)
        )

    def call(self, inputs, cache=None, use_start_token=False,
             dropout=None, attention_dropout=None,
             return_cache=False, return_logits=True, use_2d=False,
             start=None):
        """

        inputs: an integer tensor of shape [batch_size, seq_length] if not use_2d is False
                else a one_hot tensor of shape [batch_size, seq_length, vocab_size]
        cache: a list of dictionaries {"key": key, "value": value} of previous keys and values. it uses for generation
        use_start_token: if True it uses start_token for beginning.
        return_cache: if True returns new keys and values alongside output. it uses for generation.
        return_logits: if True, return logits, else return last layer embedding.
        use_2d: for tpu performances: use 2D tensors for operations and return the output in 2D shape: [batch_size * seq_length, -1]
        start: start positional embedding. if it is None, it is 0

        """
        shape = gpt2.get_tensor_shape(inputs)
        x = self.embedding(inputs, start)
        if use_start_token:
            _start_token = tf.reshape(self.start_token, [1, 1, self.embedding.embedding_size])
            _start_token = tf.tile(_start_token, [shape[0], 1, 1])
            x = tf.concat([_start_token, x], axis=1)
        if use_2d:
            shape = gpt2.get_tensor_shape(x)
            x = tf.reshape(x, [shape[0] * shape[1], shape[2]])
            shape = shape[0:2]
        else:
            shape = None
        x = self.transformer(
            inputs=x,
            cache=cache,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            use_2d=use_2d,
            shape=shape
        )
        if return_cache:
            x, cache = x
        result = dict()
        result["output"] = x
        if return_cache:
            result["cache"] = cache
        if return_logits:
            logits = tf.tensordot(x, self.embedding.word_embedding, [-1, -1])
            result["logits"] = logits
            logits = tf.tensordot(x, self.end_token, [-1, -1])
            result["end_logits"] = logits
        return result

    def __call__(self, inputs, cache=None, use_start_token=False,
                 dropout=None, attention_dropout=None,
                 return_cache=False, return_logits=True, use_2d=False,
                 start=None):
        """
        inputs: an integer tensor of shape [batch_size, seq_length] if not use_2d is False
                else a one_hot tensor of shape [batch_size, seq_length, vocab_size]
        cache: a list of dictionaries {"key": key, "value": value} of previous keys and values. it uses for generation
        use_start_token: if True it uses start_token for beginning.
        return_cache: if True returns new keys and values alongside output. it uses for generation.
        return_logits: if True, return logits, else return last layer embedding.
        use_2d: for tpu performances: use 2D tensors for operations and return the output in 2D shape: [batch_size * seq_length, -1]
        start: start positional embedding. if it is None, it is 0
        """
        return super().__call__(
            inputs=inputs,
            cache=cache,
            use_start_token=use_start_token,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            return_logits=return_logits,
            use_2d=use_2d,
            start=start
        )





class Decoder(tf.keras.Model):

    def __init__(self, config, name=None, trainable=True, dtype=None):
        super().__init__(name=name)
        self.trainable = trainable
        self.gpt2 = ModifiedGPT2(config, name="gpt2")
        self.feeder = Feeder(config, name="feeder")


    def call(self, inputs, encoded=None, cache=None,
             dropout=None, attention_dropout=None,
             return_cache=False, use_2d=False,
             encoded_length=None, start=None):
        """

        inputs: an integer tensor of shape [batch_size, seq_length] if not use_2d is False
                else a one_hot tensor of shape [batch_size, seq_length, vocab_size]
        encoded: a tensor of shape [batch_size, encoded_length, dim] if use_2d is False, else [batch_size * encoded_length, dim]
        cache: a list of dictionaries {"key": key, "value": value} of previous keys and values. it uses for generation
        return_cache: if True returns new keys and values alongside output. it uses for generation.
        use_2d: for tpu performances: use 2D tensors for operations and return the output in 2D shape: [batch_size * seq_length, -1]
        encoded_length: if use_2d then it's encoded length
        start: start positional embedding. if it is None, it is 0
        """
        if cache is None and encoded is None:
            raise ValueError("One of cache or encoded must be specified")
        if cache is not None and encoded is not None:
            raise ValueError("Only one of cache or encoded must be specified")
        if use_2d and encoded_length is None and encoded is not None:
            raise ValueError("If you are use_2d is True, then encoded_length must be specified")
        if encoded is not None:
            if use_2d:
                b = gpt2.get_tensor_shape(encoded)[0]
                shape = (b // encoded_length, encoded_length)
            else:
                shape = None
            cache = self.feeder(
                inputs=encoded,
                use_2d=use_2d,
                shape=shape
            )
        use_start_token = encoded is not None
        result = self.gpt2(
            inputs=inputs,
            cache=cache,
            use_start_token=use_start_token,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            use_2d=use_2d,
            start=start
        )
        return result

    def __call__(self, inputs, encoded=None, cache=None,
                 dropout=None, attention_dropout=None,
                 return_cache=False, use_2d=False,
                 encoded_length=None, start=None):
        """

        inputs: an integer tensor of shape [batch_size, seq_length] if not use_2d is False
                else a one_hot tensor of shape [batch_size, seq_length, vocab_size]
        encoded: a tensor of shape [batch_size, encoded_length, dim] if use_2d is False, else [batch_size * encoded_length, dim]
        cache: a list of dictionaries {"key": key, "value": value} of previous keys and values. it uses for generation
        return_cache: if True returns new keys and values alongside output. it uses for generation.
        use_2d: for tpu performances: use 2D tensors for operations and return the output in 2D shape: [batch_size * seq_length, -1]
        encoded_length: if use_2d then it's encoded length
        start: start positional embedding. if it is None, it is 0
        """
        return super().__call__(
            inputs=inputs,
            encoded=encoded,
            cache=cache,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            use_2d=use_2d,
            encoded_length=encoded_length,
            start=start
        )