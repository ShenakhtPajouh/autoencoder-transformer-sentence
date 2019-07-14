import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from modified_gpt2 import GPT2
import json
import os
import argparse

def build(config, source_weights, target_path):

    tf.enable_eager_execution()

    with open(config) as f:
        config = json.load(f)

    gpt2 = GPT2(config, "gpt2")
    encoder = Encoder(config, "encoder")
    decoder = Decoder(config, "decoder")

    inputs = tf.zeros([0, 10], tf.int32)
    length = tf.ones([0], tf.int32)
    encoded = encoder(inputs, length)
    _ = decoder(inputs, encoded)
    _ = gpt2(inputs)

    gpt2.load_weights(source_weights)

    gpt2.decoder.trainable = False
    encoder.aggregation.trainable = False
    source = gpt2.trainable_weights
    target = gpt2.trainable_weights
    for u, v in zip(source, target):
        v.assign(u)

    gpt2.decoder.trainable = True
    source = gpt2.trainable_weights
    target = [decoder.embedding.word_embedding, decoder.embedding.position_embedding]
    for block in decoder.transformer.blocks:
        block.attention.encoded_projection.trainable = False
    target = target + decoder.transformer.trainable_weights + decoder.decoder.weights
    for u, v in zip(source, target):
        v.assign(u)

    for block in decoder.transformer.blocks:
        block.attention.encoded_projection.trainable = True
    encoder.aggregation.trainable = True
    encoder_path = os.path.join(target_path, "encoder_weights.h5")
    decoder_path = os.path.join(target_path, "decoder_weights.h5")
    encoder.save_weights(encoder_path)
    decoder.save_weights(decoder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", required=True, help="config file for encoder and decocder")
    parser.add_argument("source_weights", required=True, help=".h5 file of source weights")
    parser.add_argument("target_path", required=True, help="directory for saving encoder and decoder path")
    args = parser.parse_args()
    build(args.config, args.source_weights, args.target_path)
