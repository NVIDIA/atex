# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, optimizers
from tensorflow.python.platform import tf_logging

def _m4n2_1d(matrix, patterns):
  m, n = 4, 2

  mat = tf.math.abs(tf.reshape(matrix, shape=(-1, m)))
  pmax = tf.math.argmax(tf.linalg.matmul(mat, tf.transpose(patterns)), axis=1)
  mask = tf.gather(patterns, pmax)
  mask = tf.reshape(mask, shape=matrix.shape)

  return mask

def get_2to4_mask(var, allow_padding, patterns):
  """Get a new 2:4 mask based on var.

  Conv2D stores a 4D filter weight and Dense stores a 2D kernel weight.
  For Conv2D, the filter is in the shape of (H, W, I, O) and we need to
  permute it to (H*W*O, I) and prune it along I. For Dense, the kernel is
  in shape of (I, O) and we need to permute it to (O, I) and prune it
  along I.

  Args:
    var: A weight tensor from Dense or Conv2D layers.
    allow_padding: Whether padding is allowed. Padding will be only applied to
      the input dim of Dense layers.

  Returns:
    A tensor with 2:4 mask pattern. Its shape is identical to var.
  """
  if var.shape.rank == 2:
    matrix = tf.transpose(var, perm=[1, 0])
    orig_input_dim = matrix.shape[1]
    m = 4
    padding_size = m - orig_input_dim % m
    if allow_padding and padding_size != 0:
      matrix = tf.pad(matrix, [[0, 0], [0, padding_size]], "CONSTANT")
  elif var.shape.rank == 4:
    matrix = tf.transpose(var, perm=[0, 1, 3, 2])
    permuted_shape = matrix.shape
    matrix = tf.reshape(matrix, shape=(-1, matrix.shape[-1]))

  new_mask = _m4n2_1d(matrix, patterns)

  if var.shape.rank == 2:
    if allow_padding and padding_size != 0:
      new_mask = new_mask[:, :orig_input_dim]
    new_mask = tf.transpose(new_mask, perm=[1, 0])
  elif var.shape.rank == 4:
    new_mask = tf.reshape(new_mask, shape=permuted_shape)
    new_mask = tf.transpose(new_mask, perm=[0, 1, 3, 2])

  return new_mask


