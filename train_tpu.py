import tensorflow as tf
import numpy as np
from decoder import Decoder
from encoder import Encoder
import utils
import json
import os


checkpoint_path = "checkpoints/"
train_path = "fake/train.txt"
dev_path = "fake/dev.txt"
batch_size = 64
phase = 1
EOS_Token = 4963
use_2d = True
train_steps_each_iter = 200
dev_steps_each_iter = 4

local_graph = tf.Graph()
tpu_graph = tf.Graph()

# dataset
with local_graph.as_default():
    train_data = utils.dataset(train_path, batch_size * train_steps_each_iter)
    dev_data = utils.dataset(dev_path, batch_size * dev_steps_each_iter)
    train_iter = train_data.make_one_shot_iterator()
    dev_iter = dev_data.make_one_shot_iterator()
    train_saveable = tf.data.experimental.make_saveable_from_iterator(train_iter)
    dev_saveable = tf.data.experimental.make_saveable_from_iterator(dev_iter)
    saveable = [train_saveable, dev_saveable]


# model

with open("model/hparams.json") as f:
    config = json.load(f)
with tpu_graph.as_default():
    encoder = Encoder(config, "encoder")
    decoder = Decoder(config, "decoder")
encoded_length = 1
if encoder.mode == "attention":
    encoded_length = encoder.aggregation.query_num

# optimizer
with tpu_graph.as_default():
    lr = tf.Variable(1e-4, name="lr")
    optimizer = tf.train.AdamOptimizer(lr)
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)


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
    if training:
        data = train_iter.get_next()
        tokens, length = data["tokens"], data["length"]
        tokens = tf.reshape(tokens, [8, train_steps_each_iter, batch_size // 8, -1])
        length = tf.reshape(length, [8, train_steps_each_iter, batch_size // 8])
        tokens = utils.add_EOS(tokens, length, EOS_Token)
    else:
        data = dev_iter.get_next()
        tokens, length = data["tokens"], data["length"]
        tokens = tf.reshape(tokens, [8, dev_steps_each_iter, batch_size // 8, -1])
        length = tf.reshape(length, [8, dev_steps_each_iter, batch_size // 8])
        tokens = utils.add_EOS(tokens, length, EOS_Token)
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


def train(tokens, length, training, dropout):
    training = tf.squeeze(training)
    dropout = tf.squeeze(dropout)
    tokens = tf.squeeze(tokens, 0)
    length = tf.squeeze(length, 0)
    steps = utils.get_tensor_shape(tokens)[0]
    def body(step, loss, opt):
        _tokens, _length = tf.gather(tokens, step), tf.gather(length, step)
        with tf.control_dependencies([opt]):
            _loss, opt = train_step(_tokens, _length, training, dropout)
        loss = loss + _loss
        step = step + 1
        return step, loss, opt

    def cond(step, loss, opt):
        return step < steps

    step = tf.zeros([], tf.int32)
    loss = tf.zeros([])
    opt = tf.zeros([], tf.bool)
    var_loops = [step, loss, opt]
    step, loss, opt = tf.contrib.tpu.while_loop(cond, body, var_loops)
    loss = loss / tf.cast(steps, dtype=loss.dtype)
    return loss, opt



# Graph
with tpu_graph.as_default():
    tokens_plc = tf.placeholder(tf.int32, [8, None, batch_size // 8, None])
    length_plc = tf.placeholder(tf.int32, [8, None, batch_size // 8])
    training = tf.placeholder(tf.bool, [])
    dropout = tf.placeholder(tf.float32, [])
    _training = tf.fill([8], training)
    _dropout = tf.fill([8], dropout)
    loss, opt = tf.tpu.shard(train, [tokens_plc, length_plc, _training, _dropout], 8)
    loss = tf.reduce_mean(loss, 0)
    total_loss = tf.reduce_mean(loss)
    losses = utils.str_logs(loss)
    result = {
        "total_loss": total_loss,
        "losses": losses,
        "opt": opt
    }

# run
train_logs = "logs/train.txt"
dev_logs = "logs/dev.txt"


def run(train_tokens, train_length, dev_tokens, dev_length, step):
    step = step.numpy()
    print("step:", step)
    feed_dict = {
        tokens_plc: train_tokens.numpy(),
        length_plc: train_length.numpy(),
        training: True,
        dropout: 0.0
    }
    _result = tpu_session.run(result, feed_dict)
    print("train loss:", _result["total_loss"])
    with open(train_logs, 'a') as f:
        f.write(_result["losses"].decode() + "\n")
    feed_dict = {
        tokens_plc: dev_tokens.numpy(),
        length_plc: dev_length.numpy(),
        training: False,
        dropout: 0.0
    }
    _result = tpu_session.run(result, feed_dict)
    print("dev loss:", _result["total_loss"])
    with open(dev_logs, 'a') as f:
        f.write(_result["losses"].decode() + "\n")
    return tf.zeros([], tf.bool)


# run Graph
save_steps = 20

with local_graph.as_default():
    step = tf.Variable(0, tf.int32)
    def body(step, opt):
        train_tokens, train_length = get_data(True)
        dev_tokens, dev_length = get_data(False)
        step = step + 1
        with tf.control_dependencies([opt]):
            inputs = [train_tokens, train_length, dev_tokens, dev_length, step]
            opt = tf.py_function(run, inputs, tf.bool)
            opt.set_shape([])
        return step, opt

    def cond(step, opt):
        return step < end_step

    step0 = step
    end_step = step0 + save_steps
    opt = tf.zeros([], tf.bool)
    _step, opt = tf.while_loop(cond, body, [step0, opt], parallel_iterations=2)
    _update = step.assign(_step)
    do_run = tf.group(_update, opt)

exit(0)

# Session
tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
tpu_session = tf.Session(target=tpu_address, graph=tpu_graph)
local_session = tf.Session(graph=local_graph)


# init
with tpu_graph.as_default():
    init = tf.global_variables_initializer()
    tpu_init = tf.tpu.initialize_system()
    tpu_shutdown = tf.tpu.shutdown_system()
    tpu_session.run([init, tpu_init])
    tf.keras.backend.set_session(tpu_session)
    encoder.load_weights("model/encoder_weights.h5")
    decoder.load_weights("model/decoder_weights.h5")
with local_graph.as_default():
    init = tf.global_variables_initializer()
    local_session.run(init)

# saver
var_list = get_weights() + optimizer.variables()
with tpu_graph.as_default():
    tpu_saver = utils.Saver(var_list, max_keep=100)
var_list = {v.name: v for v in saveable}
with local_graph.as_default():
    local_saver = tf.train.Saver(var_list, max_to_keep=100)

# restore
check_point_num = 0
path = checkpoint_path + "checkpoint_" + str(check_point_num) + ".pkl"
tpu_saver.restore(tpu_session, path)
path = checkpoint_path + "checkpoint_" + str(check_point_num) + ".ckpt"
local_saver.restore(path)

# lr
with tpu_graph.as_default():
    tpu_session.run(lr.assign(1e-5))

for check_point_num in range(1, 101):
    local_session.run(do_run)
    path = checkpoint_path + "checkpoint_" + str(check_point_num) + ".pkl"
    tpu_saver.save(tpu_session, path)
    path = checkpoint_path + "checkpoint_" + str(check_point_num) + ".ckpt"
    local_saver.save(path)


# save model
with tpu_graph.as_default():
    encoder.save_weights("phase1/encoder_weights.h5")
    decoder.save_weights("phase1/decoder_weights.h5")




