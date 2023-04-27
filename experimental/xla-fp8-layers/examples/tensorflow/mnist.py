"""Runs a simple mnist model with fake FP8. FP8 scaling is used.
The HLO can be dumped by setting the environment variable:
  XLA_FLAGS='--xla_dump_disable_metadata=true --xla_dump_to=/tmp/hlo'
"""
import tensorflow as tf

from fp8layers.tensorflow import Dense
from keras import layers
USE_QUANT = True


tf.keras.utils.set_random_seed(1)


# Fake FP8 dtypes since we don't yet have real FP8
from tensorflow.python.framework import dtypes
import argparse

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--fp8', action='store_true', help='use_fp8')
parser.add_argument('--bias', action='store_true', help='use_bias')
args = parser.parse_args()

use_fp8 = args.fp8
use_bias = args.bias
print("DEBUG: use_fp8", use_fp8)
print("DEBUG: use_bias", use_bias)


DenseLayer = Dense if use_fp8 else layers.Dense

class MnistModel(tf.keras.Model):

  def build(self, input_shape):
    self.dense1 = DenseLayer(64, activation="relu", use_bias=use_bias)
    self.dense2 = DenseLayer(64, activation="relu", use_bias=use_bias)
    self.dense3 = DenseLayer(16, use_bias=use_bias)


  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    output = self.dense3(x)
    return output[:,:10]

model = MnistModel()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
    # run_eagerly=True,
    jit_compile=True
)
history = model.fit(x_train, y_train, batch_size=64, epochs=20,
                    validation_split=0.0, verbose=1)

#test_scores = model.evaluate(x_test, y_test, verbose=2)
#print("Test loss:", test_scores[0])
#print("Test accuracy:", test_scores[1])
