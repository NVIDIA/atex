"""Tests for the fp8 layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import re
import tensorflow as tf

from contextlib import contextmanager
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.keras import layers, initializers, optimizers

from fp8layers.tensorflow import Dense

@contextmanager
def disable_mixed_precision(disable=True):
  try:
    if disable:
      tf.keras.mixed_precision.set_global_policy('float32')
    yield
  finally:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

def roll_and_update(amax_h, update):
  amax_h = tf.roll(amax_h, shift=-1, axis=0)
  amax_h = tf.tensor_scatter_nd_update(amax_h, [[0]], [update])
  return amax_h

def compute_scale(amax, scale, fp8_max, margin=0):
  """Default function to convert amax to scaling factor."""
  exp = tf.math.floor(tf.experimental.numpy.log2(fp8_max / amax)) - margin
  sf = tf.math.round(tf.math.pow(2., tf.math.abs(exp)))
  sf = tf.where(amax > 0.0, sf, scale)
  sf = tf.where(tf.math.is_finite(amax), sf, scale)
  sf = tf.where(exp < 0, 1.0 / sf, sf)
  return sf

class DenseTest(test.TestCase):
  def setUp(self):
    super(DenseTest, self).setUp()
    os.environ['XLA_FLAGS'] = "--xla_gpu_enable_cublaslt=true"
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

  def testDenseFwd(self):
    x = tf.random.uniform((4, 8, 16))
    init = initializers.RandomUniform(minval=0., maxval=1.)

    dense_kwargs = {
        "units": 32,
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }

    dense_ref = layers.Dense(**dense_kwargs)
    dense = Dense(**dense_kwargs)

    def _infer_step(x, model):
      return model(x)

    fn_ref = functools.partial(_infer_step, model=dense_ref)
    fn = functools.partial(_infer_step, model=dense)

    y_ref = tf.function(fn_ref, jit_compile=True)(x)
    y = tf.function(fn, jit_compile=True)(x)

    self.assertAllClose(y, y_ref, 0.05, 0.01)

  def testDenseBwd(self):
    x = tf.random.uniform(shape=(4, 8, 16))
    dy = tf.random.uniform(shape=(4, 8, 32))
    init = initializers.RandomUniform(minval=0., maxval=1.)

    dense_kwargs = {
        "units": 32,
        "use_bias": True,
        "kernel_initializer": init,
        "bias_initializer": init,
    }
    dense_ref = layers.Dense(**dense_kwargs)
    dense = Dense(**dense_kwargs)

    def _train_step(x, dy, model):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = model(x)
        loss = y * tf.cast(dy, y.dtype)
      dx, (dw, db) = tape.gradient(loss, [x, model.trainable_variables])
      return dx, dw, db, y

    fn_ref = functools.partial(_train_step, model=dense_ref)
    fn = functools.partial(_train_step, model=dense)

    dx_ref, dw_ref, db_ref, _ = tf.function(fn_ref, jit_compile=True)(x, dy)
    jit_fn = tf.function(fn, jit_compile=True)
    dx, dw, db, _ = jit_fn(x, dy)

    self.assertAllClose(dx, dx_ref, 0.05, 0.01, msg='dx tensor')
    self.assertAllClose(dw, dw_ref, 0.05, 0.01, msg='dweight tensor')
    self.assertAllClose(db, db_ref, 0.05, 0.01, msg='dbias tensor')

  def testDenseBwdHlo(self):
    x = tf.random.uniform(shape=(4, 8, 16))
    dy = tf.random.uniform(shape=(4, 8, 32))
    dense = Dense(32, use_bias=True)

    # Use the optimizer to apply the gradients to mimic the real world usage.
    optimizer = optimizers.Adam(0.01)

    @tf.function(jit_compile=True)
    def _train_step_demo(x, dy):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = dense(x)
        loss = y * tf.cast(dy, y.dtype)
      variables = dense.trainable_variables
      dx, grads = tape.gradient(loss, [x, variables])
      optimizer.apply_gradients(zip(grads, variables))
      return dx, y

    hlo = _train_step_demo.experimental_get_compiler_ir(x, dy)('optimized_hlo')
    self.assertRegex(
        hlo, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f16[32,32]{1,0}',
            'custom-call',
            'f8e4m3fn[32,16]{1,0}',
            'f8e4m3fn[32,16]{1,0}',
            'epilogue',
            'BIAS',
        )])), msg='y tensor')
    self.assertRegex(
        hlo, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f16[16,32]{1,0}',
            'custom-call',
            'f8e4m3fn[16,32]{1,0}',
            'f8e5m2[32,32]{1,0}',
        )])), msg="dx tensor")
    self.assertRegex(
        hlo, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f16[32,16]{1,0}',
            'custom-call',
            'f8e5m2[32,32]{1,0}',
            'f8e4m3fn[16,32]{1,0}',
        )])), msg="dw tensor")

  def testMatMulBias(self):
    f8e4m3 = dtypes.float8_e4m3fn
    f8e5m2 = dtypes.float8_e5m2
    E4M3_max = f8e4m3.max
    
    a_scale = tf.constant(1.0)
    b_scale = tf.constant(1.0)
    c_scale = tf.constant(1.0)
    
    @tf.function(jit_compile=True)
    def matmul_fp8(a_fp8, a_scale, b_fp8, b_scale, c_scale, bias):
        # Dequantize the inputs.
        a = tf.cast(a_fp8, in_dtype) * tf.cast(a_scale, in_dtype)
        b = tf.cast(b_fp8, in_dtype) * tf.cast(b_scale, in_dtype)
    
        if in_dtype == dtypes.float32:
          bias_cast = tf.cast(bias, dtypes.bfloat16)
          bias_cast = tf.cast(bias_cast, in_dtype)
        else:
          bias_cast = bias

        # Call the GEMM operation.
        c = tf.matmul(a, b) + bias_cast
    
        # Quantize the output.
        saturated_c = tf.clip_by_value(c / tf.cast(c_scale, in_dtype),
                                       -E4M3_max, E4M3_max)
        c_fp8 = tf.cast(saturated_c, f8e4m3)
        new_c_scale = tf.reduce_max(tf.abs(c)) / E4M3_max
    
        # Return the new scaling factors along with the results.
        # The new scaling factors will be used in the next training step (aka
        # the delayed scaling).
        return c_fp8, new_c_scale

    for in_dtype in [dtypes.float16, dtypes.float32]:
      a = tf.random.uniform((16, 64), dtype=in_dtype)
      b = tf.random.uniform((64, 16), dtype=in_dtype)
      bias = tf.random.uniform((16,), dtype=in_dtype)
      
      # Convert to FP8.
      a_fp8 = tf.cast(a, f8e4m3)
      b_fp8 = tf.cast(b, f8e4m3)
      
      hlo = matmul_fp8.experimental_get_compiler_ir(
          a_fp8, a_scale, b_fp8, b_scale, c_scale, bias)('optimized_hlo')
      self.assertRegex(
          hlo, re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f8e4m3fn[16,16]{1,0}',
              'custom-call',
              'f8e4m3fn[16,64]{1,0}',
              'f8e4m3fn[16,64]{1,0}',
              'epilogue',
              'BIAS',
          )])), msg="out tensor")

  def testMatMul(self):
    f8e4m3 = dtypes.float8_e4m3fn
    f8e5m2 = dtypes.float8_e5m2
    E4M3_max = f8e4m3.max
    
    a_scale = tf.constant(1.0)
    b_scale = tf.constant(1.0)
    c_scale = tf.constant(1.0)
    
    @tf.function(jit_compile=True)
    def matmul_fp8(a_fp8, a_scale, b_fp8, b_scale, c_scale):
        # Dequantize the inputs.
        a = tf.cast(a_fp8, in_dtype) * tf.cast(a_scale, in_dtype)
        b = tf.cast(b_fp8, in_dtype) * tf.cast(b_scale, in_dtype)
    
        # Call the GEMM operation.
        c = tf.matmul(a, b)
    
        # Quantize the output.
        saturated_c = tf.clip_by_value(c / tf.cast(c_scale, in_dtype),
                                       -E4M3_max, E4M3_max)
        c_fp8 = tf.cast(saturated_c, f8e4m3)
        new_c_scale = tf.reduce_max(tf.abs(c)) / E4M3_max
    
        # Return the new scaling factors along with the results.
        # The new scaling factors will be used in the next training step (aka
        # the delayed scaling).
        return c_fp8, new_c_scale

    for in_dtype in [dtypes.float16, dtypes.float32]:
      a = tf.random.uniform((16, 64), dtype=in_dtype)
      b = tf.random.uniform((64, 16), dtype=in_dtype)
      
      # Convert to FP8.
      a_fp8 = tf.cast(a, f8e4m3)
      b_fp8 = tf.cast(b, f8e4m3)
      
      hlo = matmul_fp8.experimental_get_compiler_ir(
          a_fp8, a_scale, b_fp8, b_scale, c_scale)('optimized_hlo')
      self.assertRegex(
          hlo, re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f8e4m3fn[16,16]{1,0}',
              'custom-call',
              'f8e4m3fn[16,64]{1,0}',
              'f8e4m3fn[16,64]{1,0}',
          )])), msg="out tensor")

  def testDenseDenseFwdHlo(self):
    for use_fp32 in [True, False]:
      with disable_mixed_precision(use_fp32):
        dtype_str = 'f32' if use_fp32 else 'f16'
        x = tf.random.uniform(shape=(4, 8, 16))

        dense_kwargs = {
            "units": 32,
            "use_bias": True,
        }
        dense0 = Dense(**dense_kwargs)
        dense1 = Dense(**dense_kwargs)

        @tf.function(jit_compile=True)
        def _infer_step(x):
          y = dense0(x)
          y = dense1(y)
          return y

        hlo = _infer_step.experimental_get_compiler_ir(x)('optimized_hlo')
        self.assertRegex(
            hlo, re.compile('.*'.join([re.escape(x) for x in (
                'cublas',
                'f8e4m3fn[32,32]{1,0}',
                'custom-call',
                'f8e4m3fn[32,16]{1,0}',
                'f8e4m3fn[32,16]{1,0}',
                'epilogue',
                'BIAS',
            )])), msg="out tensor from 1st dense")
        self.assertRegex(
            hlo, re.compile('.*'.join([re.escape(x) for x in (
                'cublas',
                dtype_str,
                '[32,32]{1,0}',
                'custom-call',
                'f8e4m3fn[32,32]{1,0}',
                'f8e4m3fn[32,32]{1,0}',
            )])), msg="out tensor from 2nd dense")

  def testDenseDenseBwdHlo(self):
    for use_fp32 in [True, False]:
      with disable_mixed_precision(use_fp32):
        dtype_str = 'f32' if use_fp32 else 'f16'
        x = tf.random.uniform(shape=(4, 8, 16))
        dy = tf.random.uniform(shape=(4, 8, 32))

        # In the bprop, the dy (=dx of the 2nd dense) will be used to compute
        # the dbias of the 1st dense in higher precision. This will force the dy
        # to be in higher precison. To test the fp8 output, we need to disable
        # the bias.
        dense0 = Dense(32, use_bias=False)
        dense1 = Dense(32)

        # Use the optimizer to apply the gradients to mimic the real world case.
        optimizer = optimizers.Adam(0.01)

        @tf.function(jit_compile=True)
        def _train_step(x, dy):
          with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = dense0(x)
            y = dense1(y)
            loss = y * tf.cast(dy, y.dtype)

          variables = dense0.trainable_variables + dense1.trainable_variables

          dx, grads = tape.gradient(loss, [x, variables])
          optimizer.apply_gradients(zip(grads, variables))
          return dx, y

        hlo = _train_step.experimental_get_compiler_ir(x, dy)('optimized_hlo')
        self.assertRegex(
            hlo, re.compile('.*'.join([re.escape(x) for x in (
                'cublas',
                'f8e5m2[32,32]{1,0}',
                'custom-call',
                'f8e5m2[32,32]{1,0}',
                'f8e4m3fn[32,32]{1,0}',
            )])), msg="dx tensor from 2nd dense")
        self.assertRegex(
            hlo, re.compile('.*'.join([re.escape(x) for x in (
                'cublas',
                dtype_str,
                '[16,32]{1,0}',
                'custom-call',
                'f8e4m3fn[16,32]{1,0}',
                'f8e5m2[32,32]{1,0}',
            )])), msg="dw tensor from 1st dense")
        self.assertRegex(
            hlo, re.compile('.*'.join([re.escape(x) for x in (
                'cublas',
                dtype_str,
                '[32,16]{1,0}',
                'custom-call',
                'f8e5m2[32,32]{1,0}',
                'f8e4m3fn[16,32]{1,0}',
          )])), msg="dx tensor from 1st dense")
        self.assertRegex(
            hlo, re.compile('.*'.join([re.escape(x) for x in (
                'cublas',
                'f8e4m3fn[32,32]{1,0}',
                'custom-call',
                'f8e4m3fn[32,16]{1,0}',
                'f8e4m3fn[32,16]{1,0}',
            )])), msg="out tensor from 1st dense")
        self.assertRegex(
            hlo, re.compile('.*'.join([re.escape(x) for x in (
                'cublas',
                dtype_str,
                '[32,32]{1,0}',
                'custom-call',
                'f8e4m3fn[32,32]{1,0}',
                'f8e4m3fn[32,32]{1,0}',
                'epilogue',
                'BIAS',
            )])), msg="out tensor from 2nd dense")      
        self.assertRegex(
            hlo, re.compile('.*'.join([re.escape(x) for x in (
                'cublas',
                dtype_str,
                '[32,32]{1,0}',
                'custom-call',
                'f8e4m3fn[32,32]{1,0}',
                'f8e5m2[32,32]{1,0}',
           )])), msg="dw tensor from 2nd dense")

  def testDenseFwdAmaxBookkeeping(self):

    dense_kwargs = {
        "units": 32,
        "use_bias": True,
        "amax_history_length": 3,
    }
    dense = Dense(**dense_kwargs)

    @tf.function(jit_compile=True)
    def _train_step(x, dy):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = dense(x)
        loss = y * tf.cast(dy, y.dtype)
      dx, grads = tape.gradient(loss, [x, dense.trainable_variables])
      return dx, grads, y

    optimizer = optimizers.Adam(0.01)

    amax_history_x = tf.zeros((dense.amax_history_length,))
    amax_history_k = tf.zeros((dense.amax_history_length,))
    amax_history_dy = tf.zeros((dense.amax_history_length,))
    scale_x = tf.ones(())
    scale_k = tf.ones(())
    scale_dy = tf.ones(())

    for _ in range(5):
      x = tf.random.normal(shape=(4, 8, 16))
      dy = tf.random.normal(shape=(4, 8, 32))

      _, grads, _ = _train_step(x, dy)
      amax_history_x = roll_and_update(amax_history_x, tf.reduce_max(tf.abs(x)))
      amax_history_k = roll_and_update(
          amax_history_k, tf.reduce_max(tf.abs(dense.kernel)))
      amax_history_dy = roll_and_update(
          amax_history_dy, tf.reduce_max(tf.abs(dy)))

      rtol, atol = 0.001, 0.001
      self.assertAllClose(dense.input_amax_history, amax_history_x, rtol, atol)
      self.assertAllClose(dense.kernel_amax_history, amax_history_k, rtol, atol)
      self.assertAllClose(dense.output_grad_amax_history, amax_history_dy, rtol,
                          atol)

      amax_from_history_x = tf.reduce_max(amax_history_x, axis=0)
      amax_from_history_k = tf.reduce_max(amax_history_k, axis=0)
      amax_from_history_dy = tf.reduce_max(amax_history_dy, axis=0)
      scale_x = compute_scale(
          amax_from_history_x, scale_x, dtypes.float8_e4m3fn.max)
      scale_k = compute_scale(
          amax_from_history_k, scale_k, dtypes.float8_e4m3fn.max)
      scale_dy = compute_scale(
          amax_from_history_dy, scale_dy, dtypes.float8_e5m2.max)

      self.assertAllClose(1. / dense.input_scale, scale_x)
      self.assertAllClose(1. / dense.kernel_scale, scale_k)
      self.assertAllClose(1. / dense.output_grad_scale, scale_dy)

      optimizer.apply_gradients(zip(grads, dense.trainable_variables))


if __name__ == '__main__':
    test.main()

