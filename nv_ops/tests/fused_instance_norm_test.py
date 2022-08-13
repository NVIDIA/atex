# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

"""Tests for fused instance norm ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

from nv_norms import InstanceNormalization
from nv_norms import fused_instance_norm_op, fused_instance_norm_grad_op


def instance_norm_grad_np(x, dy, gamma, cache, is_channel_first):
  N_axis = (0, )
  if is_channel_first:
    D_axis = tuple([i for i in range(2, x.ndim)])
    C_axis = (1, )
  else:
    D_axis = tuple([i for i in range(1, x.ndim-1)])
    C_axis = (-1, )

  ND_axis = N_axis + D_axis

  D = 1
  for dim in D_axis:
    D *= x.shape[dim]

  istd = cache["istd"]
  mean = cache["mean"]

  expand_d = -1 if is_channel_first else 1
  expand_g = -1 if is_channel_first else 0

  for i in range(len(D_axis)):
    istd = np.expand_dims(istd, expand_d)
    mean = np.expand_dims(mean, expand_d)
    gamma = np.expand_dims(gamma, expand_g)
  gamma = np.expand_dims(gamma, 0)

  x_mean = x - mean

  dgamma = np.sum(dy * x_mean * istd, axis=ND_axis, dtype=np.float32)
  dbeta = np.sum(dy, axis=ND_axis, dtype=np.float32)

  dl_di = dy * gamma * istd
  di_dx = 1.

  dl_dvar = np.sum(dy * gamma * x_mean * (-0.5) * (istd**3), axis=D_axis,
                   keepdims=True,  dtype=np.float32)
  dvar_dx = 2. * x_mean / D

  dl_dmean = np.sum(-1. * dy * gamma * istd, axis=D_axis, keepdims=True, 
                    dtype=np.float32)
  dmean_dx = 1. / D

  dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmean * dmean_dx
  return dgamma, dbeta, dx

def get_input_shape(N, C, D, x_rank, axis):
  assert axis in (1, -1)
  x_shape = [N]
  if axis == 1:
    x_shape += [C] + [D] * (x_rank - 2)
  else:
    x_shape += [D] * (x_rank - 2) + [C]
  return x_shape

class FusedInstanceNormOpTest(test.TestCase):
  def _runForward(self, x_shape, axis, epsilon=0.001):
    assert axis in (1, -1)
    x = tf.random.normal(shape=x_shape, stddev=10.0, dtype=tf.float32)
    gamma = tf.constant(
        np.random.normal(size=x_shape[axis]),
        dtype=tf.float32)
    beta = tf.constant(
        np.random.normal(size=x_shape[axis]),
        dtype=tf.float32)
    instanceN_ref = tfa.layers.InstanceNormalization(
        axis=axis, center=True, scale=True, epsilon=epsilon)
    instanceN_ref.build(input_shape=x_shape)
    instanceN_ref.set_weights([gamma, beta])
    y_ref = instanceN_ref(x)
    if axis == 1:
      reduce_axis = tuple([i for i in range(2, x.ndim)])
    else:
      reduce_axis = tuple([i for i in range(1, x.ndim-1)])

    mean_ref, var_ref = tf.nn.moments(x, axes=reduce_axis)
    inv_var_ref = tf.constant(1. / (var_ref + epsilon),  dtype=tf.float32)
    
    # For ops fused_instance_norm_op, fused_instance_norm_grad_op, they take
    # argument data_format in ("NC...", "N...C", "NCHW", "NHWC", "NCDHW", 
    # "NDHWC")
    op_data_format = "NC..." if axis == 1 else "N...C"
    y, mean, inv_std = fused_instance_norm_op(
        x, gamma, beta, epsilon=epsilon, data_format=op_data_format)
    self.assertAllClose(y, y_ref, atol=0.01)
    self.assertAllClose(mean, mean_ref, atol=0.01)
    self.assertAllClose(inv_std**2, inv_var_ref, atol=0.05)

  def _runBackward(self, x_shape, axis, epsilon=0.0):
    assert axis in (1, -1)
    x_np = np.random.normal(0.0, 10.0, size=x_shape).astype(np.float32)
    dy_np = np.random.normal(size=x_shape).astype(np.float32)
    gamma_np = np.random.normal(size=x_shape[axis]).astype(np.float32)

    x = tf.constant(x_np, dtype=tf.float32)
    dy = tf.constant(dy_np, dtype=tf.float32)
    gamma = tf.constant(gamma_np, dtype=tf.float32)

    if axis == 1:
      reduce_axis = tuple([i for i in range(2, x.ndim)])
    else:
      reduce_axis = tuple([i for i in range(1, x.ndim-1)])

    mean, var = tf.nn.moments(x, axes=reduce_axis)
    
    inv_std = tf.constant(1. / np.sqrt(var + epsilon), dtype=tf.float32)
    cache = {}
    cache["istd"] = inv_std
    cache["mean"] = mean

    grad_op_data_format = "NC..." if axis == 1 else "N...C"
    dx, dgamma, dbeta = fused_instance_norm_grad_op(
        dy, x, gamma, mean, inv_std, data_format=grad_op_data_format)

    dgamma_ref, dbeta_ref, dx_ref = instance_norm_grad_np(
        x_np, dy_np, gamma_np, cache, axis == 1)

    self.assertAllClose(dx_ref, dx, atol=0.02)
    self.assertAllClose(dbeta_ref, dbeta, atol=0.05)
    self.assertAllClose(dgamma_ref, dgamma, atol=0.05)

  @test_util.run_gpu_only
  def testFusedInstanceNormOp(self):
    N, C = 2, 32
    with self.cached_session(use_gpu=True) as sess:
      for x_rank in (4, 5):
        for D_exp in (1, 2, 3, 4, 5, 6,):
          for axis in (1, -1):
            x_shape = get_input_shape(N, C, 2**D_exp, x_rank, axis)
            self._runForward(x_shape, axis)
            self._runBackward(x_shape, axis)
  
  #4 ,1, 1 forward on volta
  @test_util.run_gpu_only
  def testFusedInstanceNormEmptyInput(self):
    with self.cached_session(use_gpu=True) as sess:
      x = tf.constant([], dtype=tf.float32)
      x = tf.reshape(x, shape=(0, 0, 0, 0, 0))
      gamma = tf.constant([], dtype=tf.float32)
      beta = tf.constant([], dtype=tf.float32)

      data_format = "NC..."
      y, mean, inv_var = fused_instance_norm_op(
          x, gamma, beta, data_format=data_format)
      self.assertAllEqual(y.shape, [0, 0, 0, 0, 0])
      self.assertAllEqual(mean.shape, [0, 0])
      self.assertAllEqual(inv_var.shape, [0, 0])

  @test_util.run_gpu_only
  def testFusedInstanceNormGradEmptyInput(self):
    with self.cached_session(use_gpu=True) as sess:
      dy = tf.constant([], dtype=tf.float32)
      dy = tf.reshape(dy, shape=(0, 0, 0, 0, 0))
      x = tf.constant([], dtype=tf.float32)
      x = tf.reshape(x, shape=(0, 0, 0, 0, 0))
      gamma = tf.constant([], dtype=tf.float32)
      mean = tf.constant([], dtype=tf.float32)
      inv_var = tf.constant([], dtype=tf.float32)
      mean = tf.reshape(mean, shape=(0, 0))
      inv_var = tf.reshape(inv_var, shape=(0, 0))

      data_format = "N...C"
      dx, dgamma, dbeta = fused_instance_norm_grad_op(
          dy, x, gamma, mean, inv_var, data_format=data_format)
      self.assertAllEqual(dx.shape, [0, 0, 0, 0, 0])
      self.assertAllEqual(dgamma.shape, [0])
      self.assertAllEqual(dbeta.shape, [0])

class FusedInstanceNormLayerTest(test.TestCase):
  def _runForward(self, x_shape, axis, epsilon=0.001):
    assert axis in (1, -1)
    x = tf.constant(np.random.normal(size=x_shape), dtype=tf.float32)
    gamma = tf.constant(
        np.random.normal(size=x_shape[axis]),
        dtype=tf.float32)
    beta = tf.constant(
        np.random.normal(size=x_shape[axis]),
        dtype=tf.float32)
    instanceN_ref = tfa.layers.InstanceNormalization(
        axis=axis, center=True, scale=True, epsilon=epsilon)
    instanceN_ref.build(input_shape=x_shape)
    instanceN_ref.set_weights([gamma, beta])
    y_ref = instanceN_ref(x)

    instanceN = InstanceNormalization(axis=axis)
    instanceN.build(input_shape=x_shape)
    instanceN.set_weights([gamma, beta])
    y = instanceN(x)

    self.assertAllClose(y, y_ref, rtol=0.01, atol=0.01)

  def _runBackward(self, x_shape, axis, epsilon=0.01):
    assert axis in (1, -1)
    x = tf.constant(np.random.normal(size=x_shape), dtype=tf.float16)
    gamma = tf.constant(np.random.normal(size=x_shape[axis]),
                        dtype=tf.float32)
    beta = tf.constant(np.random.normal(size=x_shape[axis]),
                       dtype=tf.float32)

    instanceN = InstanceNormalization(axis=axis)
    instanceN.build(input_shape=x_shape)
    instanceN.set_weights([gamma, beta])

    instanceN_ref = tfa.layers.InstanceNormalization(axis=axis)
    instanceN_ref.build(input_shape=x_shape)
    instanceN_ref.set_weights([gamma, beta])
    y_true = tf.random.normal(shape=x_shape)

    def get_grads(instanceN):
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = instanceN(x)
        loss = tf.math.reduce_mean(
            tf.keras.losses.binary_crossentropy(y, y_true))

      dx, (dgamma, dbeta) = tape.gradient(loss, [x, instanceN.variables])
      return dx, dgamma, dbeta

    dx_ref, dgamma_ref, dbeta_ref = get_grads(instanceN_ref)
    dx, dgamma, dbeta = get_grads(instanceN)

    self.assertAllClose(dx_ref, dx, atol=0.05)
    self.assertAllClose(dbeta_ref, dbeta, atol=0.05)
    self.assertAllClose(dgamma_ref, dgamma, atol=0.05)

  @test_util.run_gpu_only
  def testFusedInstanceNorm(self):
    N, C = 2, 32
    with self.cached_session(use_gpu=True) as sess:
      rank, features = 5, 4
      axes = [-1, 1]
      for axis in axes:
        x_shape = get_input_shape(N, C, 2**features, rank, axis)
        self._runForward(x_shape, axis)
        self._runBackward(x_shape, axis)


if __name__ == '__main__':
  test.main()

