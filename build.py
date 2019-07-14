import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from modified_gpt2 import GPT2
import json

tf.enable_eager_execution()

with open("source/hparams.json") as f:
    config = json.load(f)

gpt2 = GPT2(config, "gpt2")
encoder = Encoder(config, "encoder")
decoder = Decoder(config, "decoder")

inputs = tf.zeros([0, 10], tf.int32)
length = tf.ones([0], tf.int32)
encoded = encoder(inputs, length)
_ = decoder(inputs, encoded)
_ = gpt2(inputs)

gpt2.load_weights("source/weights.h5")

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

encoder.save_weights("model/encoder_weights.h5")
decoder.save_weights("model/decoder_weights.h5")
