# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================
import nv_norms
import tensorflow as tf
import tensorflow_addons as tfa
import time

from tensorflow.keras import mixed_precision

def benchmark_fn(input_shape, use_nv_ops, axis=-1):
  mixed_precision.set_global_policy('mixed_float16')
  warmup = 10
  repeat = 20
  instanceN = tfa.layers.InstanceNormalization(axis=axis)
  if use_nv_ops:
    instanceN = nv_norms.InstanceNormalization(axis=axis)

  def train_step(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
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
def get_shape(x, channel_last):
  if channel_last:
    return (x[0], x[2], x[2], x[2], x[1])
  else:
    return (x[0], x[1], x[2], x[2], x[2])

for input_shape in input_shapes:
  expanded_shape = get_shape(input_shape, True)
  time_tf = benchmark_fn(expanded_shape, False, axis=-1)
  time_nv = benchmark_fn(expanded_shape, True, axis=-1)
  print("Input: {} Time(ms): TF: {:0.2f} NV: {:0.2f}".format(
      expanded_shape, time_tf, time_nv))
print("End of channel last layout.")

for input_shape in input_shapes:
  expanded_shape = get_shape(input_shape, False)
  time_tf = benchmark_fn(expanded_shape, False, axis=1)
  time_nv = benchmark_fn(expanded_shape, True, axis=1)
  print("Input: {} Time(ms): TF: {:0.2f} NV: {:0.2f}".format(
      expanded_shape, time_tf, time_nv))
print("End of channel first layout.")

