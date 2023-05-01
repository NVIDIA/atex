# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================
import argparse
from atex import nv_norms
import tensorflow as tf
import tensorflow_addons as tfa
import time

from tensorflow.keras import mixed_precision

parser = argparse.ArgumentParser(description='Benchmark configs')
parser.add_argument('--xla', action='store_true', help='Use XLA for reference')
args = parser.parse_args()

use_xla = args.xla

def train_step_func(x, instanceN):
  with tf.GradientTape() as tape:
    tape.watch(x)
    y = instanceN(x)
    loss = tf.reduce_sum(y)
  dx, (dgamma, dbeta) = tape.gradient(loss, [x, instanceN.variables])
  return dx, dgamma, dbeta

def benchmark_fn(input_shape, use_nvops, axis):
  mixed_precision.set_global_policy('mixed_float16')
  warmup = 10
  repeat = 20

  train_step = train_step_func
  if use_nvops:
    instanceN = nv_norms.InstanceNormalization(axis=axis)
  else:
    instanceN = tfa.layers.InstanceNormalization(axis=axis)
    if use_xla:
      train_step = tf.function(train_step, jit_compile=True)

  instanceN.build(input_shape)

  data = tf.random.normal(input_shape)

  for i in range(warmup):
    dx, dgamma, dbeta = train_step(data, instanceN)

  _ = tf.reduce_sum(dx).numpy()

  start = time.time()
  for i in range(repeat):
    dx, dgamma, dbeta = train_step(data, instanceN)

  _ = tf.reduce_sum(dx).numpy()

  result = time.time() - start
  return 1000 * result / repeat


# denote N C D/H/W dim
input_shapes = [
    (2, 32, 6),
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
