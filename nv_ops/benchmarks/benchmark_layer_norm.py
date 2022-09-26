# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================
import nv_norms
import tensorflow as tf
import time

from tensorflow.keras import mixed_precision

def benchmark_fn(input_shape, use_nv_ops):
  mixed_precision.set_global_policy('mixed_float16')
  warmup = 10
  repeat = 100

  layerN = tf.keras.layers.LayerNormalization(axis=(1,))
  if use_nv_ops:
    layerN = nv_norms.LayerNormalization(axis=(1,))

  def train_step(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = layerN(x)
      loss = tf.reduce_sum(y)
    dx, (dgamma, dbeta) = tape.gradient(loss, [x, layerN.variables])
    return dx, dgamma, dbeta

  data = tf.random.normal(input_shape)
  if use_nv_ops:
    # We manually cast the input to fp16. In practice, however, the output of
    # the previous layer should already be fp16 when using 'mixed_float16'.
    data = tf.cast(data, tf.float16)

  for i in range(warmup):
    dx, dgamma, dbeta = train_step(data)
  _ = tf.reduce_sum(dx).numpy()

  start = time.time()
  for i in range(repeat):
    dx, dgamma, dbeta = train_step(data)
  _ = tf.reduce_sum(dx).numpy()

  result = time.time() - start
  return 1000 * result / repeat


input_shapes = [
    (10, 10000000),
    (100, 1000000),
    (1000, 100000),
    (10000, 10000),
    (100000, 1000),
    (1000000, 100),
    (10000000, 10),
    (4, 400001), # Non-typical shapes
    (4, 10000001),
]

for input_shape in input_shapes:
  assert len(input_shape) == 2
  time_tf = benchmark_fn(input_shape, False)
  time_nv = benchmark_fn(input_shape, True)
  print("Input: {} {} Time(ms): TF: {:0.2f} NV: {:0.2f}".format(
      input_shape[0], input_shape[1], time_tf, time_nv))
