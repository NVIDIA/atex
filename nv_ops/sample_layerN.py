# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

import argparse
import nv_norms
import tensorflow as tf
from tensorflow.keras import layers, models

parser = argparse.ArgumentParser(description="Use --nvops to replace LayerN")
parser.add_argument('--nvops', action='store_true',
                    help="""Whether to Fused Layer Norm.""")
args, _ = parser.parse_known_args()

N, H, W, C = (10, 3, 3, 4)
k, c, r, s = (4, C, 2, 2)
use_nv_norms = True if args.nvops else False

conv2d = layers.Conv2D(k, (r, s), padding='same')
layerN = layers.LayerNormalization(axis=(1,2,3))

if use_nv_norms:
  # Call the build() to create weights.
  layerN.build((N, H, W, k))

def model():
  x = layers.Input(shape=(H, W, C), batch_size=None)
  y = conv2d(x)
  if use_nv_norms:
    z, _, _ = nv_norms.fused_layer_norm(y, layerN.gamma, layerN.beta)
  else:
    z = layerN(y)
  return models.Model(x, z, name='toy_model')

toy_model = model()

@tf.function
def train_step(x):
  with tf.GradientTape() as tape:
    y = toy_model(x)
    loss = tf.reduce_sum(y)
  if use_nv_norms:
    # The weights in layerN are no longer tracked in the toy_model.
    grads = tape.gradient(loss, [toy_model.variables, layerN.variables])
  else:
    grads = tape.gradient(loss, [toy_model.variables])
  return grads

data = tf.random.normal((N, H, W, C))
g = train_step(data)

_ = g[0][0].numpy() # sync GPU
print("Done with", "Fused LayerN" if use_nv_norms else "Keras LayerN")

