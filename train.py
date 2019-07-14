import tensorflow as tf
import numpy as np
from decoder import Decoder
from encoder import Encoder
import utils
import json
import time

checkpoint_path = "checkpoints/"
train_path = "fake/train.txt"
dev_path = "fake/dev.txt"
batch_size = 64
phase = 1
EOS_Token = 4963
use_2d = False


# dataset

train_data = utils.dataset(train_path, batch_size)
dev_data = utils.dataset(dev_path, batch_size)
train_iter = train_data.make_one_shot_iterator()
dev_iter = dev_data.make_one_shot_iterator()
train_saveable = tf.data.experimental.make_saveable_from_iterator(train_iter)
dev_saveable = tf.data.experimental.make_saveable_from_iterator(dev_iter)
saveables = [train_saveable, dev_saveable]


# model

with open("model/hparams.json") as f:
    config = json.load(f)
encoder = Encoder(config, "encoder")
decoder = Decoder(config, "decoder")
encoded_length = 1
if encoder.mode == "attention":
    encoded_length = encoder.aggregation.query_num

# optimizer
lr = tf.Variable(1e-4, name="lr")
optimizer = tf.train.AdamOptimizer(lr)


def get_weights():
    weights = []
    if phase == 1:
        if encoder.mode == "attention":
            weights = weights + encoder.aggregation.trainable_weights
        weights.append(decoder.embedding.start_embedding)
        for block in decoder.transformer.blocks:
            weights = weights + block.attention.encoded_projection.trainable_weights
    elif phase == 2:
        weights = encoder.trainable_weights + decoder.trainable_weights
    return weights


def get_data(training):

    def true_fn():
        data = train_iter.get_next()
        tokens, length = data["tokens"], data["length"]
        tokens = utils.add_EOS(tokens, length, EOS_Token)
        return tokens, length

    def false_fn():
        data = train_iter.get_next()
        tokens, length = data["tokens"], data["length"]
        tokens = utils.add_EOS(tokens, length, EOS_Token)
        return tokens, length

    tokens, length = tf.cond(training, true_fn, false_fn)
    return tokens, length


def train_step(tokens, length, training, dropout):
    inputs = tokens[:, :-1]
    encoded = encoder(
        inputs=inputs,
        length=length,
        dropout=dropout,
        attention_dropout=dropout,
        use_2d=use_2d
    )
    logits = decoder(
        inputs=inputs,
        encoded=encoded,
        dropout=dropout,
        attention_dropout=dropout,
        use_2d=use_2d,
        encoded_length=encoded_length
    )
    ln = utils.get_tensor_shape(tokens)[1]
    mask = tf.sequence_mask(length + 1, ln)
    loss = utils.loss(tokens, logits, mask, use_2d=use_2d)

    def true_fn():
        weights = get_weights()
        grads = tf.gradients(loss, weights)
        opt = optimizer.apply_gradients(zip(grads, weights))
        with tf.control_dependencies([opt]):
            opt = tf.zeros([], tf.bool)
        return opt

    def false_fn():
        opt = tf.zeros([], tf.bool)
        return opt

    opt = tf.cond(training, true_fn, false_fn)
    return loss, opt


def train(steps, training, dropout):

    def body(step, loss, opt):
        tokens, length = get_data(training)
        with tf.control_dependencies([opt]):
            _loss, opt = train_step(tokens, length, training, dropout)
        _loss = tf.expand_dims(_loss, 0)
        loss = tf.concat([loss, _loss], 0)
        step = step + 1
        return step, loss, opt

    def cond(step, loss, opt):
        return step < steps

    step = tf.zeros([], tf.int32)
    loss = tf.zeros([0])
    opt = tf.zeros([], tf.bool)
    var_loops = [step, loss, opt]
    invariant_shape = [tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([])]
    step, loss, opt = tf.while_loop(cond, body, var_loops, invariant_shape, parallel_iterations=2)
    total_loss = tf.reduce_mean(loss)
    losses = utils.str_logs(loss)
    result = {
        "total_loss": total_loss,
        "losses": losses,
        "opt": opt
    }
    return result


# Graph

steps = tf.placeholder(tf.int32, [])
training = tf.placeholder(tf.bool, [])
dropout = tf.placeholder(tf.float32, [])
result = train(steps, training, dropout)

exit(0)

# Session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session()


# init
init = tf.global_variables_initializer()
session.run(init)
tf.keras.backend.set_session(session)
encoder.load_weights("model/encoder_weights.h5")
decoder.load_weights("model/decoder_weights.h5")


# saver
var_list = get_weights() + optimizer.variables() + saveables
var_list = {v.name: v for v in var_list}
saver = tf.train.Saver(var_list, max_to_keep=100)

# restore
check_point_num = 0
path = checkpoint_path + "checkpoint_" + str(check_point_num) + ".ckpt"
saver.restore(session, path)

# lr
session.run(lr.assign(1e-5))

# run
total_steps = 100
step = 0
check_point_num = 0
train_logs = "logs/train.txt"
dev_logs = "logs/dev.txt"
while step < total_steps:
    step += 1
    feed_dict = {
        steps: 200,
        training: True,
        dropout: 0.0
    }
    _result = session.run(result, feed_dict)
    print("step:", step)
    print("train loss:", _result["total_loss"])
    with open(train_logs, 'a') as f:
        f.write(_result["losses"].decode() + "\n")
    feed_dict = {
        steps: 4,
        training: False,
        dropout: 0.0
    }
    _result = session.run(result, feed_dict)
    print("dev loss:", _result["total_loss"])
    with open(dev_logs, 'a') as f:
        f.write(_result["losses"].decode() + "\n")
    if step % 10 == 0:
        check_point_num += 1
        path = checkpoint_path + "checkpoint_" + str(check_point_num) + ".ckpt"
        saver.save(session, path)
