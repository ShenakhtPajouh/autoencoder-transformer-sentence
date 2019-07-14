# Sentence Auto-Encoder using gpt2 structure

## Example for using Auto-encoder:

#### Traning
In training you will pass the result as well as encoded vectors to decoder.

```python
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
import json
import utils

with open("config/file") as f:
    config = json.load(f)
encoder = Encoder(config, name="encoder")
decoder = Decoder(config, name="decoder")

tokens = tf.placeholder(tf.int32, [32, 50])       # [batch_size, seq_length]
length = tf.placeholder(tf.int32, [32])           # length of each sample
encoded = encoder(tokens, length)                 # return a [batch_size, num_vector, dim] tensor.
                                                  # num_vector is the number of vectors which we encode are sentence.
                                                  # for instance [32, 2, 768]
EOS_char = 4964                                   # the last token
inputs = utils.add_EOS(tokens, length, EOS_char)  # add EOS_char to the end of inputs 
predict = decoder(inputs=inputs, encoded=encoded) # return [batch_size, seq_length + 1, dim] for logits
                                                  # labels = inputs in training
```

#### Generation
generation only works in Eager Exectution mode. use cache in order to generate a sequence from an encoded vectors:

```python
def top_k_sampling(logits, k=20, tempt=0.5):  # logits is a single vector for instance [4964]
    logits, indices = tf.math.top_k(logits, k)
    logits = logits / tempt
    logits = tf.expand_dims(logits, 0)
    i = tf.random.categorical(logits, 1)
    i = tf.squeeze(i)
    return indices[i]


encoded = tf.random.normal([2, 768])  # suppose it is an encoded vector for a single sentence.
encoded = tf.expand_dims(encoded, 0)  # [1, 2, 768], in order to pass decoder model
tokens = []
EOS_token = 4964


inputs = tf.convert_to_tensor([[]], tf.int32)
logits, cache = decoder(inputs=inputs, encoded=encoded, return_cache=True)
logits = tf.squeeze(logits, 0)
_token = top_k_sampling(logits).numpy()
while _token != EOS_token:
    tokens.append(_token)
    inputs = tf.convert_to_tensor([[_token]], tf.int32)
    logits, cache = decoder(inputs=inputs, cache=cache, return_cache=True)
    logits = tf.squeeze(logits, 0)
    _token = top_k_sampling(logits).numpy()

print(tokens)  # tokens will be the decoded sentence.
```