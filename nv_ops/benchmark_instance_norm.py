# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

import nv_norms
import tensorflow as tf
import tensorflow_addons as tfa
import time

from tensorflow.keras import mixed_precision

def benchmark_fn(input_shape, use_nv_ops, data_format="NC..."):
  mixed_precision.set_global_policy('mixed_float16')
  manual_mixed_float16 = True
  warmup = 10
  repeat = 20
  channel_axis = -1 if data_format == "N...C" else 1
  gamma, beta = None, None
  instanceN = tfa.layers.InstanceNormalization(axis=channel_axis)
  if use_nv_ops:
    # Call the build() to create weights.
    instanceN.build(input_shape=input_shape)
    gamma, beta = instanceN.weights[0], instanceN.weights[1]

  def train_step(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      if use_nv_ops:
        y, _, _ = nv_norms.fused_instance_norm(
            x, gamma, beta, data_format=data_format)
      else:
        y = instanceN(x)
      loss = tf.reduce_sum(y)
    dx, (dgamma, dbeta) = tape.gradient(loss, [x, instanceN.variables])
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


# denote N C D/H/W dim
input_shapes = [
    (2, 32, 128),
    (2, 64, 128),
    (4, 32, 128),
    (4, 64, 64),
    (8, 32, 64),
    (8, 64, 64),
    (8, 128, 64),
    (4, 256, 32),
    (8, 256, 32),
]
for input_shape in input_shapes:
  def sw(x): return (x[0], x[2], x[2], x[2], x[1])
  expanded_shape = sw(input_shape)
  time_tf = benchmark_fn(expanded_shape, False, "N...C")
  time_nv = benchmark_fn(expanded_shape, True, "N...C")
  print("Input: {} Time(ms): TF: {:0.2f} NV: {:0.2f}".format(
      expanded_shape, time_tf, time_nv))

print("End of NHWC")
for input_shape in input_shapes:
  def sw(x): return (x[0], x[1], x[2], x[2], x[2])
  expanded_shape = sw(input_shape)
  time_tf = benchmark_fn(expanded_shape, False, "NC...")
  time_nv = benchmark_fn(expanded_shape, True, "NC...")
  print("Input: {} Time(ms): TF: {:0.2f} NV: {:0.2f}".format(
      expanded_shape, time_tf, time_nv))
print("End of NCHW")
