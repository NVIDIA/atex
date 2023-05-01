# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

"""Tests for fused layer norm ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

from nv_norms import fused_layer_norm_op, fused_layer_norm_grad_op
from nv_norms import LayerNormalization


def layer_norm_grad_np(x, dy, gamma, cache, axis):
  assert x.ndim >= 2, "x and dy have to be larger than 1D."
  gamma_shape = gamma.shape
  x_shape = x.shape

  D = 1
  for a in axis:
    D *= x_shape[a]
  N = x.size // D

  x = x.reshape((N, D))
  dy = dy.reshape((N, D))
  gamma = gamma.reshape((D, ))

  N_axis = (0, )
  D_axis = (1, )
  N = x.shape[0]
  D = x.shape[1]

  istd = cache["istd"].numpy()
  mean = cache['mean'].numpy()

  # We manually expand istd and mean from (N,) to (N,1) to facilitate the
  # broadcasting in the following computation.
  mean = np.expand_dims(mean, -1)
  istd = np.expand_dims(istd, -1)

  x_mean = x - mean
  dgamma = np.sum(dy * x_mean * istd, axis=N_axis)
  dbeta = np.sum(dy, axis=N_axis)
  dl_di = dy * gamma * istd
  di_dx = 1.
  dl_dvar = np.sum(dy * gamma * x_mean * (-0.5) * (istd**3), axis=D_axis,
                   keepdims=True)
  dvar_dx = 2. * x_mean / D
  dl_dmean = np.sum(-1. * dy * gamma * istd, axis=D_axis, keepdims=True)
  dmean_dx = 1. / D
  dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmean * dmean_dx

  dgamma = dgamma.reshape(gamma_shape)
  dbeta = dbeta.reshape(gamma_shape)
  dx = dx.reshape(x_shape)
  return dgamma, dbeta, dx

class NvNormsLayerNormOpTest(test.TestCase):
  def _runForward(self, x_shape, data_dtype, axis, epsilon=0.001):
    validated_axis = sorted(set([i % len(x_shape) for i in axis]))
    weight_shape = [x_shape[i] for i in validated_axis]

    x = tf.random.normal(shape=x_shape, stddev=10.0, dtype=tf.float32)                          
    gamma = tf.constant(np.random.normal(size=weight_shape), dtype=tf.float32)
    beta = tf.constant(np.random.normal(size=weight_shape), dtype=tf.float32)
    ref_ln = tf.keras.layers.LayerNormalization(
        axis=validated_axis, center=True, scale=True, epsilon=epsilon)
    ref_ln.build(input_shape=x_shape)
    ref_ln.set_weights([gamma, beta])
    y_ref = ref_ln(x)
    mean_ref, var_ref = tf.nn.moments(x, axes=validated_axis)
    inv_var_ref = tf.constant(1. / (var_ref + epsilon),  dtype=tf.float32)
    mean_ref = tf.reshape(mean_ref, shape=-1)
    var_ref = tf.reshape(var_ref, shape=-1)
    y, mean, inv_std = fused_layer_norm_op(x, gamma, beta, axis=axis)
    self.assertAllClose(y, y_ref, rtol=0.01, atol=0.01)
    self.assertAllClose(mean, mean_ref, rtol=0.01, atol=0.01)
    self.assertAllClose(inv_std**2, inv_var_ref, rtol=0.01, atol=0.01)

  def _runBackward(self, x_shape, data_dtype, axis, epsilon=0.001):
    validated_axis = sorted(set([i % len(x_shape) for i in axis]))
    weight_shape = [x_shape[i] for i in validated_axis]

    x_np = np.random.normal(0.0, 10.0, size=x_shape)
    dy_np = np.random.normal(size=x_shape)
    gamma_np = np.random.normal(size=weight_shape)

    x = tf.constant(x_np, dtype=data_dtype)
    dy = tf.constant(dy_np, dtype=data_dtype)
    gamma = tf.constant(gamma_np, dtype=tf.float32)
    mean, var = tf.nn.moments(x, axes=validated_axis)

    mean = tf.reshape(mean, shape=-1)
    var = tf.reshape(var, shape=-1)

    inv_std = tf.constant(1. / np.sqrt(var + epsilon), dtype=tf.float32)
    cache = {}
    cache["istd"] = tf.cast(inv_std, tf.float32)
    cache["mean"] = tf.cast(mean, tf.float32)

    dx, dgamma, dbeta = fused_layer_norm_grad_op(
        dy, x, gamma, cache["mean"], cache["istd"], axis=axis)
    dgamma_ref, dbeta_ref, dx_ref = layer_norm_grad_np(
        x_np, dy_np, gamma_np, cache, axis=validated_axis)
    self.assertAllClose(dx_ref, dx, rtol=0.01, atol=0.08)
    self.assertAllClose(dbeta_ref, dbeta, rtol=0.01, atol=0.01)
    self.assertAllClose(dgamma_ref, dgamma, rtol=0.02, atol=0.02)
  
  @test_util.run_gpu_only
  def testFusedLayerNormOp(self):
    with self.cached_session(use_gpu=True):
      dtypes = [tf.float32, tf.float16]
      ranks = [2, 3]
      batches = [1, 2, 5, 8]
      features = [4, 8, 10, 15, 18, 19]
      for dtype, rank, N, D in itertools.product(dtypes, ranks, batches,
                                                 features):
        axis = [-1] if rank == 2 else [-2, -1]
        x_shape = [N] * (rank - 1)
        x_shape.append(2**D)
        self._runForward(x_shape, dtype, axis)
        self._runBackward(x_shape, dtype, axis)
  
  @test_util.run_gpu_only
  def testFusedLayerNormOpWithNonTypicalInputShapes(self):
    with self.cached_session(use_gpu=True):
      dtypes = [tf.float16, tf.float32]
      N = 2
      features = [11, 12, 31, 2003, 4001, 5002, 2**14 + 1, 2**16 + 2, 2**18 + 3]
      for dtype, D in itertools.product(dtypes, features):
        x_shape = [N, D]
        self._runForward(x_shape, dtype, [-1])
        self._runBackward(x_shape, dtype, [-1])

  @test_util.run_gpu_only
  def testFusedLayerNormOpWithBrittleShapes(self):
    with self.cached_session(use_gpu=True):
      x_shapes = [
          [12000, 128, 8, 8],
          [1200, 128, 8, 6],
          [2, 128, 8, 8],
          [1, 64, 384, 276]]
      for x_shape in x_shapes:
        self._runForward(x_shape, tf.float32, [1, 2, 3])
        self._runBackward(x_shape, tf.float32, [1, 2, 3])

  @test_util.run_gpu_only
  def testFusedLayerNormEmptyInput(self):
    with self.cached_session(use_gpu=True):
      x = tf.constant([], dtype=tf.float32)
      x = tf.reshape(x, shape=(0, 0))
      gamma = tf.constant([], dtype=tf.float32)
      beta = tf.constant([], dtype=tf.float32)
      y, mean, inv_var = fused_layer_norm_op(x, gamma, beta)
      self.assertAllEqual(y.shape, [0, 0])
      self.assertAllEqual(mean.shape, [0])
      self.assertAllEqual(inv_var.shape, [0])

  @test_util.run_gpu_only
  def testFusedLayerNormGradEmptyInput(self):
    with self.cached_session(use_gpu=True):
      dy = tf.constant([], dtype=tf.float32)
      dy = tf.reshape(dy, shape=(0, 0))
      x = tf.constant([], dtype=tf.float32)
      x = tf.reshape(x, shape=(0, 0))
      gamma = tf.constant([], dtype=tf.float32)
      mean = tf.constant([], dtype=tf.float32)
      inv_var = tf.constant([], dtype=tf.float32)
      dx, dgamma, dbeta = fused_layer_norm_grad_op(
          dy, x, gamma, mean, inv_var)
      self.assertAllEqual(dx.shape, [0, 0])
      self.assertAllEqual(dgamma.shape, [0])
      self.assertAllEqual(dbeta.shape, [0])

class NvNormsLayerNormLayerTest(test.TestCase):
  def _runForward(self, x_shape, data_dtype, axis, epsilon=0.001):
    if isinstance(axis, int):
      weight_shape = x_shape[axis]
    else:
      weight_shape = [x_shape[i] for i in axis]
    x = tf.random.uniform(shape=x_shape, minval=10.0,
                          maxval=1000.0, dtype=data_dtype)
    gamma = tf.constant(np.random.normal(size=weight_shape), dtype=tf.float32)
    beta = tf.constant(np.random.normal(size=weight_shape), dtype=tf.float32)
    layerN_ref = tf.keras.layers.LayerNormalization(
        axis=axis, center=True, scale=True, epsilon=epsilon)
    layerN_ref.build(input_shape=x_shape)
    layerN_ref.set_weights([gamma, beta])

    y_ref = layerN_ref(x)
    layerN = LayerNormalization(axis=axis)
    layerN.build(input_shape=x_shape)
    layerN.set_weights([gamma, beta])
    y = layerN(x)

    self.assertAllClose(y, y_ref, rtol=0.01, atol=0.01)

  def _runBackward(self, x_shape, data_dtype, axis):
    if isinstance(axis, int):
      weight_shape = x_shape[axis]
    else:
      weight_shape = [x_shape[i] for i in axis]
    x = tf.constant(np.random.normal(size=x_shape), dtype=data_dtype)
    gamma = tf.constant(np.random.normal(size=weight_shape), dtype=tf.float32)
    beta = tf.constant(np.random.normal(size=weight_shape), dtype=tf.float32)

    layerN = LayerNormalization(axis=axis)
    layerN.build(input_shape=x_shape)
    layerN.set_weights([gamma, beta])

    layerN_ref = tf.keras.layers.LayerNormalization(
        axis=axis, center=True, scale=True)
    layerN_ref.build(input_shape=x_shape)
    layerN_ref.set_weights([gamma, beta])

    def get_grads(layerN):
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = layerN(x)
      dx, (dgamma, dbeta) = tape.gradient(y, [x, layerN.variables])
      return dx, dgamma, dbeta

    dx, dgamma, dbeta = get_grads(layerN)
    dx_ref, dgamma_ref, dbeta_ref = get_grads(layerN_ref)

    self.assertAllClose(dx_ref, dx, rtol=0.01, atol=0.01)
    self.assertAllClose(dbeta_ref, dbeta, rtol=0.01, atol=0.01)
    self.assertAllClose(dgamma_ref, dgamma, rtol=0.02, atol=0.02)

  @test_util.run_gpu_only
  def testFusedLayerNorm(self):
    with self.cached_session(use_gpu=True):
      dtypes = [tf.float32, tf.float16]
      rank, N, D = 3, 8, 8
      for dtype in dtypes:
        axis = [-1] if rank == 2 else [-2, -1]
        x_shape = [N] * (rank - 1)
        x_shape.append(2 ** D)
        self._runForward(x_shape, dtype, axis)
        self._runBackward(x_shape, dtype, axis)

  @test_util.run_gpu_only
  def testFusedLayerNormWithDifferentAxis(self):
    axes = [[1, 2], [-2, -1], [1, -1], [-2, 2], -1]
    for axis in axes:
      self._runForward([2, 3, 4], tf.float32, axis)
      self._runBackward([2, 3, 4], tf.float32, axis)

  @test_util.run_gpu_only
  def testLayerWithIntegerAxis(self):
    axes = [-1, 2]
    for axis in axes:
      self._runForward([2, 3, 4], tf.float32, axis)
      self._runBackward([2, 3, 4], tf.float32, axis)

if __name__ == '__main__':
  test.main()

