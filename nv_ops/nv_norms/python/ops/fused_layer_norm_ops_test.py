# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

"""Tests for time_two ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

try:
  from nv_norms.python.ops import nv_norm_ops
except ImportError:
  import nv_norm_ops

def layer_norm_grad_np(x, dy, gamma, cache):
  N_axis = (0, )
  D_axis = tuple([i for i in range(1, x.ndim)])
  N = x.shape[0]
  D = x.size / N

  ivar = cache["ivar"].numpy()
  mean = cache['mean'].numpy()

  for _ in range(len(D_axis)):
    mean = np.expand_dims(mean, -1)
    ivar = np.expand_dims(ivar, -1)
  x_mean = x - mean
  dgamma = np.sum(dy * x_mean * ivar, axis=N_axis)
  dbeta = np.sum(dy, axis=N_axis)
  dl_di = dy * gamma * ivar
  di_dx = 1.
  dl_dvar = np.sum(dy * gamma * x_mean * (-0.5) * (ivar**3), axis=D_axis,
                   keepdims=True)
  dvar_dx = 2. * x_mean / D
  dl_dmean = np.sum(-1. * dy * gamma * ivar, axis=D_axis, keepdims=True)
  dmean_dx = 1. / D
  dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmean * dmean_dx
  return dgamma, dbeta, dx

class FusedLayerNormTest(test.TestCase):
  def _runForward(self, x_shape, epsilon=0.001):
    feature_dim = [i for i in range(1, len(x_shape))]
    x = tf.random.uniform(shape=x_shape, minval=10.0,
                          maxval=1000.0, dtype=tf.float16)
    gamma = tf.constant(np.random.normal(size=x_shape[1:]), dtype=tf.float32)
    beta = tf.constant(np.random.normal(size=x_shape[1:]), dtype=tf.float32)
    ref_ln = tf.keras.layers.LayerNormalization(axis=feature_dim,
                                                center=True, scale=True,
                                                epsilon=epsilon)
    ref_ln.build(input_shape=x_shape)
    ref_ln.set_weights([gamma, beta])
    y_ref = ref_ln(x)
    mean_ref, var_ref = tf.nn.moments(x, axes=feature_dim)
    y, mean, inv_var = nv_norm_ops.fused_layer_norm(x, gamma, beta)
    self.assertAllClose(y, y_ref, rtol=0.01, atol=0.01)
    self.assertAllClose(mean, mean_ref, rtol=0.01, atol=0.01)
    self.assertAllClose(inv_var, 1./var_ref, rtol=0.01, atol=0.01)

  def _runBackward(self, x_shape, epsilon=0.001):
    feature_dim = tuple(range(1, len(x_shape)))
    x_np = np.random.normal(size=x_shape)
    dy_np = np.random.normal(size=x_shape)
    gamma_np = np.random.normal(size=x_shape[1:])

    x = tf.constant(x_np, dtype=tf.float16)
    dy = tf.constant(dy_np, dtype=tf.float16)
    gamma = tf.constant(gamma_np, dtype=tf.float32)
    mean, var = tf.nn.moments(x, axes=feature_dim)
    inv_var = 1. / var
    cache = {}
    cache["ivar"] = tf.cast(inv_var, tf.float32)
    cache["mean"] = tf.cast(mean, tf.float32)

    dx, dgamma, dbeta = nv_norm_ops.fused_layer_norm_grad(
        dy, x, gamma, cache["mean"], cache["ivar"])
    dgamma_ref, dbeta_ref, dx_ref = layer_norm_grad_np(
        x_np, dy_np, gamma_np, cache)
    self.assertAllClose(dx_ref, dx, rtol=0.01, atol=0.01)
    self.assertAllClose(dbeta_ref, dbeta, rtol=0.01, atol=0.01)
    self.assertAllClose(dgamma_ref, dgamma, rtol=0.01, atol=0.01)

  @test_util.run_gpu_only
  def testFusedLayerNorm(self):
    with self.cached_session(use_gpu=True) as sess:
      for x_rank in (2, 3,):
        for N in (2, 8,):
          for D_exp in (4, 8, 10, 18, 19):
            x_shape = [N] * (x_rank - 1)
            x_shape.append(2 ** D_exp)
            self._runForward(x_shape)
            self._runBackward(x_shape)
      
  @test_util.run_gpu_only
  def testFusedLayerNormEmptyInput(self):
    with self.cached_session(use_gpu=True) as sess:
      x = tf.constant([], dtype=tf.float32)
      x = tf.reshape(x, shape=(0, 0))
      gamma = tf.constant([], dtype=tf.float32)
      beta = tf.constant([], dtype=tf.float32)
      y, mean, inv_var = nv_norm_ops.fused_layer_norm(x, gamma, beta)
      self.assertAllEqual(y.shape, [0, 0])
      self.assertAllEqual(mean.shape, [0])
      self.assertAllEqual(inv_var.shape, [0])

  @test_util.run_gpu_only
  def testFusedLayerNormGradEmptyInput(self):
    with self.cached_session(use_gpu=True) as sess:
      dy = tf.constant([], dtype=tf.float32)
      dy = tf.reshape(dy, shape=(0, 0))
      x = tf.constant([], dtype=tf.float32)
      x = tf.reshape(x, shape=(0, 0))
      gamma = tf.constant([], dtype=tf.float32)
      mean = tf.constant([], dtype=tf.float32)
      inv_var = tf.constant([], dtype=tf.float32)
      dx, dgamma, dbeta = nv_norm_ops.fused_layer_norm_grad(
          dy, x, gamma, mean, inv_var)
      self.assertAllEqual(dx.shape, [0, 0])
      self.assertAllEqual(dgamma.shape, [0])
      self.assertAllEqual(dbeta.shape, [0])


if __name__ == '__main__':
  test.main()
