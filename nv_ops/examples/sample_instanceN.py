# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

import argparse
import nv_norms
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models

parser = argparse.ArgumentParser(description="Use --nvops to replace InstanceN")
parser.add_argument('--nvops', action='store_true',
                    help="""Whether to Fused Instance Norm.""")
args, _ = parser.parse_known_args()

N, H, W, C = (2, 32, 32, 8)
k, c, r, s = (4, C, 2, 2)
use_nv_norms = True if args.nvops else False
axis = -1
conv2d = layers.Conv2D(k, (r, s), padding='same')
instanceN = tfa.layers.InstanceNormalization(axis=axis)
if use_nv_norms:
  instanceN = nv_norms.InstanceNormalization(axis=axis)

def model():
  x = layers.Input(shape=(H, W, C), batch_size=None)
  y = conv2d(x)
  z = instanceN(y)
  return models.Model(x, z, name='toy_model')

toy_model = model()

@tf.function
def train_step(x):
  with tf.GradientTape() as tape:
    y = toy_model(x)
    loss = tf.reduce_sum(y)
  if use_nv_norms:
    # The weights in instanceN are no longer tracked in the toy_model.
    grads = tape.gradient(loss, [toy_model.variables, instanceN.variables])
  else:
    grads = tape.gradient(loss, [toy_model.variables])
  return grads

data = tf.random.normal((N, H, W, C))
g = train_step(data)

_ = g[0][0].numpy() # sync GPU
print("Done with", "Fused instanceN" if use_nv_norms else "tfa instanceN")

