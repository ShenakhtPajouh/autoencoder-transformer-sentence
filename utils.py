import tensorflow as tf
import pickle
import os


def get_tensor_shape(x):
    x = tf.convert_to_tensor(x)
    static_shape = x.shape.as_list()
    if tf.executing_eagerly():
        return static_shape
    dynamic_shape = tf.shape(x)
    if static_shape is None:
        return dynamic_shape
    dynamic_shape = tf.unstack(dynamic_shape)
    shape = []
    for st, dyn in zip(static_shape, dynamic_shape):
        if st is None:
            shape.append(dyn)
        else:
            shape.append(st)
    return shape


def dataset(path, batch_size):
    data = tf.data.TextLineDataset(path)
    data = data.repeat()
    data = data.map(map)
    output_shapes = {"tokens": tf.TensorShape([None]), "length": tf.TensorShape([])}
    data = data.padded_batch(batch_size, output_shapes, drop_remainder=True)
    return data


def map(text):
    text = tf.expand_dims(text, 0)
    tokens = tf.strings.split(text, ' ').values
    tokens = tf.string_to_number(tokens, tf.int32)
    length = tf.shape(tokens)[0]
    return {"tokens": tokens, "length": length}


def loss(labels, logits, mask, use_2d=False):
    if use_2d:
        mask = tf.reshape(mask, [-1])
        labels = tf.reshape(labels, [-1])
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss = loss * tf.cast(mask, loss.dtype)
    loss = tf.reduce_mean(loss)
    return loss

def add_EOS(tokens, length, EOS_char):
    shape = get_tensor_shape(tokens)
    seq_ln = shape[-1]
    _padd_labels = tf.zeros(shape=shape[:-1] + [1], dtype=tokens.dtype)
    tokens = tf.concat([tokens, _padd_labels], axis=-1)
    EOS = tf.one_hot(length, depth=seq_ln + 1, dtype=tokens.dtype) * EOS_char
    tokens = tokens + EOS
    return tokens



def str_logs(logs):
    logs = tf.as_string(logs)
    logs = tf.strings.reduce_join(logs, separator=' ')
    return logs


class Saver(object):
    def __init__(self, var_list=None, max_keep=10):
        self._var_list = var_list
        self.max_keep = max_keep
        self.saved = []
        if self._var_list is None:
            self._var_list = tf.global_variables()
        self._var_plc = [tf.placeholder(v.dtype, v.shape) for v in self._var_list]
        self._ops = [v.assign(u) for u, v in zip(self._var_plc, self._var_list)]

    def save(self, session, address):
        vars_list = session.run(self._var_list)
        with open(address, "wb") as f:
            pickle.dump(vars_list, f)
        self.saved.append(address)
        if len(self.saved) > self.max_keep:
            del_address = self.saved[0]
            self.saved = self.saved[1:]
            os.remove(del_address)

    def restore(self, session, address):
        with open(address, "rb") as f:
            var_list = pickle.load(f)
        feed_dict = {u: v for u, v in zip(self._var_plc, var_list)}
        _ = session.run(self._ops, feed_dict)
