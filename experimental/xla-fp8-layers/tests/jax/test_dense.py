"""Tests for the fp8 layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from functools import partial
import re

import optax

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
from jax import lax
from jax import random

import flax
from flax import linen as nn
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict

from fp8layers.jax import DenseGeneral, TrainState

def roll_and_update(amax_h, update):
  return jnp.roll(amax_h, shift=-1, axis=0).at[0].set(update)

def compute_scale(amax, scale, fp8_max, margin=0):
  """Default function to convert amax to scaling factor."""
  exp = jnp.floor(jnp.log2(fp8_max / amax)) - margin
  sf = jnp.round(lax.pow(2., jnp.abs(exp)))
  sf = jnp.where(amax > 0.0, sf, scale)
  sf = jnp.where(lax.is_finite(amax), sf, scale)
  sf = jnp.where(exp < 0, 1.0 / sf, sf)
  return sf

@jtu.with_config(jax_numpy_rank_promotion='allow',
                 jax_numpy_dtype_promotion='standard')
class DenseTest(jtu.JaxTestCase):

  def testDenseFwd(self):
    x = random.uniform(random.PRNGKey(1), (48, 16))
    init = nn.initializers.uniform(1.)

    dense_kwargs = {
        "features": 32,
        "use_bias": True,
        "kernel_init": init,
        "bias_init": init,
        "dtype": jnp.bfloat16,
    }

    dense_ref = nn.Dense(**dense_kwargs)
    dense = DenseGeneral(**dense_kwargs)
    
    key = random.PRNGKey(0)
    variables_ref = dense_ref.init(key, x)
    variables = dense.init(key, x)

    def _infer(model, variables, x):
      y = model.apply(variables, x)
      return y

    infer_fn_ref = jax.jit(partial(_infer, dense_ref))
    y_ref = infer_fn_ref(variables_ref, x)

    infer_fn = jax.jit(partial(_infer, dense))
    y = infer_fn(variables, x)

    self.assertAllClose(y, y_ref, atol=0.1, rtol=0.05)

  def testDenseBwd(self) :
    x = random.uniform(random.PRNGKey(1), (16, 16))
    dy = random.uniform(random.PRNGKey(1), (16, 32))
    init = nn.initializers.uniform(1.)

    dense_kwargs = {
        "features": 32,
        "use_bias": True,
        "kernel_init": init,
        "bias_init": init,
        "dtype": jnp.bfloat16,
    }

    dense_ref = nn.Dense(**dense_kwargs)
    dense = DenseGeneral(**dense_kwargs)

    key = random.PRNGKey(0)
    variables_ref = dense_ref.init(key, x)
    variables = dense.init(key, x)

    def _train_loss(model, variables, x):
      y = model.apply(variables, x)
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)

    train_fn_ref = jax.jit(jax.value_and_grad(partial(_train_loss, dense_ref),
                                              argnums=[0, 1]))
    loss_val_ref, grads_ref = train_fn_ref(variables_ref, x)

    train_fn = jax.jit(jax.value_and_grad(partial(_train_loss, dense),
                                          argnums=[0, 1]))
    loss_val, grads = train_fn(variables, x)

    db_ref = grads_ref[0]['params']['bias']
    dw_ref = grads_ref[0]['params']['kernel']
    dx_ref = grads_ref[1]

    db = grads[0]['params']['bias']
    dw = grads[0]['params']['kernel']
    dx = grads[1]

    self.assertAllClose(dw_ref, dw, atol=0.05, rtol=0.01)
    self.assertAllClose(db_ref, db, atol=0.05, rtol=0.01)
    self.assertAllClose(dx_ref, dx, atol=0.05, rtol=0.01)

  def testDenseShape(self):
    x = random.uniform(random.PRNGKey(1), (4, 8, 8, 16))
    init = nn.initializers.uniform(1.)

    dense_kwargs = {
        "features": [16, 4],
        "axis": [-2, -1],
        "use_bias": True,
        "kernel_init": init,
        "bias_init": init,
        "dtype": jnp.bfloat16,
    }

    dense = DenseGeneral(**dense_kwargs)
    
    key = random.PRNGKey(0)
    variables = dense.init(key, x)

    def _infer(model, variables, x):
      y = model.apply(variables, x)
      return y

    infer_fn = jax.jit(partial(_infer, dense))
    y = infer_fn(variables, x)
    self.assertEqual(y.shape, (4, 8, 16, 4))

  def testDenseBwdHlo(self):
    in_dtype = jnp.float32
    key = random.PRNGKey(1)
    x = random.uniform(key, (16, 16), dtype=in_dtype)
    dy = random.uniform(key, (16, 32), dtype=in_dtype)

    dense = DenseGeneral(32, dtype=jnp.bfloat16)

    @jax.jit
    def _train_step(params, x_data):
      def loss_fn(params, x_data):
        y = dense.apply(params, x_data)
        loss = y * dy.astype(y.dtype)
        return jnp.mean(loss)

      loss_grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1])
      loss, gradients = loss_grad_fn(params, x_data)
      return loss, gradients

    variables = dense.init(key, jnp.ones((16, 16)))

    hlo = _train_step.lower(variables, x).compile()
    self.assertRegex(
        hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[16,32]{1,0}',
            'custom-call',
            'f8e4m3fn[16,16]{1,0}',
            'f8e4m3fn[32,16]{1,0}',
            'epilogue',
            'BIAS',
        )])), msg='y tensor')
    self.assertRegex(
        hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[16,32]{1,0}',
            'custom-call',
            'f8e4m3fn[16,16]{1,0}',
            'f8e5m2[32,16]{1,0}',
        )])), msg="dx tensor")
    self.assertRegex(
        hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[16,16]{1,0}',
            'custom-call',
            'f8e5m2[16,32]{1,0}',
            'f8e4m3fn[16,32]{1,0}',
        )])), msg="dw tensor")

  def testMatMulBias(self):
    f8e4m3 = jnp.float8_e4m3fn
    f8e5m2 = jnp.float8_e5m2
    key = random.PRNGKey(42)
    
    a_scale = 1.
    b_scale = 1.
    c_scale = 1.
    
    @jax.jit
    def matmul_fp8(a_fp8, a_scale, b_fp8, b_scale, c_scale, bias):
        # Dequantize the inputs.
        a = a_fp8.astype(in_dtype) * a_scale.astype(in_dtype)
        b = b_fp8.astype(in_dtype) * b_scale.astype(in_dtype)
    
        if in_dtype == jnp.float32:
          bias_cast = bias.astype(jnp.bfloat16)
          bias_cast = bias_cast.astype(in_dtype)
        else:
          bias_cast = bias

        # Call the GEMM operation.
        c = jax.lax.dot(a, b) + bias_cast
    
        # Quantize the output.
        E4M3_max = jnp.finfo(f8e4m3).max.astype(in_dtype)
        saturated_c = jnp.clip(c / c_scale.astype(in_dtype), -E4M3_max,
                               E4M3_max)
        c_fp8 = saturated_c.astype(f8e4m3)
        new_c_scale = jnp.max(jnp.abs(c)).astype(in_dtype) / E4M3_max
    
        # Return the new scaling factors along with the results.
        # The new scaling factors will be used in the next training step (aka
        # the delayed scaling).
        return c_fp8, new_c_scale

    for in_dtype in [jnp.float16, jnp.float32, jnp.bfloat16]:
      a = random.uniform(key, (16, 16), dtype=in_dtype)
      b = random.uniform(key, (16, 16), dtype=in_dtype)
      bias = random.uniform(key, (16,), dtype=in_dtype)
      
      # Convert to FP8.
      a_fp8 = a.astype(f8e4m3)
      b_fp8 = b.astype(f8e4m3)  
  
      hlo = jax.jit(matmul_fp8).lower(
          a_fp8, a_scale, b_fp8, b_scale, c_scale, bias).compile()
      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f8e4m3fn[16,16]{1,0}',
              'custom-call',
              'f8e4m3fn[16,16]{1,0}',
              'f8e4m3fn[16,16]{1,0}',
              'epilogue',
              'BIAS',
          )])), msg='out tensor')

  def testMatMul(self):
    f8e4m3 = jnp.float8_e4m3fn
    f8e5m2 = jnp.float8_e5m2
    key = random.PRNGKey(42)
    
    a_scale = 1.
    b_scale = 1.
    c_scale = 1.
    
    @jax.jit
    def matmul_fp8(a_fp8, a_scale, b_fp8, b_scale, c_scale):
        # Dequantize the inputs.
        a = a_fp8.astype(in_dtype) * a_scale.astype(in_dtype)
        b = b_fp8.astype(in_dtype) * b_scale.astype(in_dtype)
    
        # Call the GEMM operation.
        c = jax.lax.dot(a, b)
    
        # Quantize the output.
        E4M3_max = jnp.finfo(f8e4m3).max.astype(in_dtype)
        saturated_c = jnp.clip(c / c_scale.astype(in_dtype), -E4M3_max,
                               E4M3_max)
        c_fp8 = saturated_c.astype(f8e4m3)
        new_c_scale = jnp.max(jnp.abs(c)).astype(in_dtype) / E4M3_max
    
        # Return the new scaling factors along with the results.
        # The new scaling factors will be used in the next training step (aka
        # the delayed scaling).
        return c_fp8, new_c_scale

    for in_dtype in [jnp.float32, jnp.bfloat16]:
      a = random.uniform(key, (16, 16), dtype=in_dtype)
      b = random.uniform(key, (16, 16), dtype=in_dtype)
      
      # Convert to FP8.
      a_fp8 = a.astype(f8e4m3)
      b_fp8 = b.astype(f8e4m3)  
  
      hlo = jax.jit(matmul_fp8).lower(
          a_fp8, a_scale, b_fp8, b_scale, c_scale).compile()
      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f8e4m3fn[16,16]{1,0}',
              'custom-call',
              'f8e4m3fn[16,16]{1,0}',
              'f8e4m3fn[16,16]{1,0}',
          )])), msg='out tensor')

  def testDenseDenseFwdHlo(self):
    for use_fp32 in [False, True]:
      dtype = jnp.float32 if use_fp32 else jnp.bfloat16
      dtype_str = 'f32' if use_fp32 else 'bf16'

      key = random.PRNGKey(1)
      x = random.uniform(key, (16, 16))

      class DenseDense(nn.Module):
        def setup(self):
          self.dense1 = DenseGeneral(32, dtype=dtype)
          self.dense2 = DenseGeneral(16, dtype=dtype)
        
        def __call__(self, inputs):
          x = self.dense1(inputs)
          x = self.dense2(x)
          return x

      # Initialize the parameters of the neural network
      net = DenseDense()
      variables = net.init(key, x)
      
      # NOTE: init has to be outside and captured by jitted function.
      # Otherwise, no bias will be seen.
      def predict(params, x_data):
        return net.apply(params, x_data)

      hlo = jax.jit(predict).lower(variables, x).compile()
      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f8e4m3fn[16,32]{1,0}',
              'custom-call',
              'f8e4m3fn[16,16]{1,0}',
              'f8e4m3fn[32,16]{1,0}',
              'epilogue',
              'BIAS',
          )])), msg='out tensorfrom 1st dense')
      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              dtype_str,
              '[16,16]{1,0}',
              'custom-call',
              'f8e4m3fn[16,32]{1,0}',
              'f8e4m3fn[16,32]{1,0}',
              'epilogue',
              'BIAS',
          )])), msg='out tensor from 2nd dense')
  
  def testDenseDenseBwdHlo(self):
    for use_fp32 in [True, False]:
      dtype = jnp.float32 if use_fp32 else jnp.bfloat16
      dtype_str = 'f32' if use_fp32 else 'bf16'

      key = random.PRNGKey(1)
      x = random.uniform(key, (16, 16))
      dy = random.uniform(key, (16, 32))

      class DenseDense(nn.Module):
        def setup(self):
          # When the bwd pass is turned on, the dy (dx of the second dense) will
          # be used for the dbias of the first dense. Because of this, the output
          # will be in higher precision. Therefore, we disable the bias.
          self.dense1 = DenseGeneral(32, use_bias=False, dtype=dtype)
          self.dense2 = DenseGeneral(32, dtype=dtype)
        
        def __call__(self, inputs):
          x = self.dense1(inputs)
          x = self.dense2(x)
          return x

      model = DenseDense()

      def _train_step(params, x_data):
        def loss_fn(params, x_data):
          y = model.apply(params, x_data)
          loss = y * dy.astype(y.dtype)
          return jnp.mean(loss)

        loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=[0, 1]))
        loss, gradients = loss_grad_fn(params, x_data)
        return loss, gradients

      variables = model.init(key, jnp.ones((16, 16)))
      _, grad = _train_step(variables, x)

      hlo = jax.jit(_train_step).lower(variables, x).compile()

      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f8e4m3fn[16,32]{1,0}',
              'custom-call',
              'f8e4m3fn[16,16]{1,0}',
              'f8e4m3fn[32,16]{1,0}',
          )])), msg="out tensor from 1st dense")
      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              dtype_str,
              '[16,32]{1,0}',
              'custom-call',
              'f8e4m3fn[16,32]{1,0}',
              'f8e4m3fn[32,32]{1,0}',
              'epilogue',
              'BIAS',
          )])), msg="out tensor from 2nd dense")
      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f8e5m2[16,32]{1,0}',
              'custom-call',
              'f8e5m2[16,32]{1,0}',
              'f8e4m3fn[32,32]{1,0}',
          )])), msg="dx tensor from 2nd dense")
      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              dtype_str,
              '[16,32]{1,0}',
              'custom-call',
              'f8e4m3fn[16,16]{1,0}',
              'f8e5m2[32,16]{1,0}',
          )])), msg="dw tensor from 2nd dense")
      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              dtype_str,
              '[32,32]{1,0}',
              'custom-call',
              'f8e4m3fn[32,16]{1,0}',
              'f8e5m2[32,16]{1,0}',
          )])), msg="dx tensor from 1st dense")
      self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              dtype_str,
              '[16,16]{1,0}',
              'custom-call',
              'f8e5m2[16,32]{1,0}',
              'f8e4m3fn[16,32]{1,0}',
          )])), msg="dw tensor from 1st dense")

  def testDenseBwdAmaxBookkeeping(self):
    x = random.uniform(random.PRNGKey(1), (16, 16), dtype=jnp.float32)

    dense_kwargs = {
        "features": 32,
        "use_bias": True,
        "amax_history_length": 3,
    }
    dense = DenseGeneral(**dense_kwargs)

    key = random.PRNGKey(0)
    variables = dense.init(key, x)

    opt = optax.adam(learning_rate=.1)
    state = TrainState.create(model_variables=variables, tx=opt, apply_fn=None)

    def _train_loss(variables, x, dy):
      y = dense.apply(variables, x)
      loss = y * dy.astype(y.dtype)
      return jnp.sum(loss)

    train_fn = jax.jit(jax.value_and_grad(_train_loss, argnums=[0]))

    amax_history_x = jnp.zeros((dense.amax_history_length,))
    amax_history_k = jnp.zeros((dense.amax_history_length,))
    amax_history_dy = jnp.zeros((dense.amax_history_length,))
    scale_x = jnp.ones(())
    scale_k = jnp.ones(())
    scale_dy = jnp.ones(())
    fp8_e4m3_max = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)
    fp8_e5m2_max = jnp.finfo(jnp.float8_e5m2).max.astype(jnp.float32)

    for _ in range(5):
      x = random.normal(random.PRNGKey(1), (16, 16), dtype=jnp.float32)
      dy = random.normal(random.PRNGKey(1), (16, 32), dtype=jnp.float32)

      loss_val, grads = train_fn(state.variables(), x, dy)

      amax_history_x = roll_and_update(amax_history_x, jnp.max(jnp.abs(x)))
      amax_history_k = roll_and_update(
          amax_history_k, jnp.max(jnp.abs(state.params['kernel'])))
      amax_history_dy = roll_and_update(amax_history_dy, jnp.max(jnp.abs(dy)))

      amax_from_history_x = jnp.max(amax_history_x, axis=0)
      amax_from_history_k = jnp.max(amax_history_k, axis=0)
      amax_from_history_dy = jnp.max(amax_history_dy, axis=0)
      scale_x = compute_scale(amax_from_history_x, scale_x, fp8_e4m3_max)
      scale_k = compute_scale(amax_from_history_k, scale_k, fp8_e4m3_max)
      scale_dy = compute_scale(amax_from_history_dy, scale_dy, fp8_e5m2_max)

      state = state.apply_gradients(grads=grads[0])

      rtol, atol = 0.001, 0.001
      fp8_vars = state.fp8_params
      self.assertAllClose(fp8_vars['input_amax_history_fp8_meta'],
                          amax_history_x, rtol=rtol, atol=atol)
      self.assertAllClose(fp8_vars['kernel_amax_history_fp8_meta'],
                          amax_history_k, rtol=rtol, atol=atol)
      self.assertAllClose(fp8_vars['output_grad_amax_history_fp8_meta'],
                          amax_history_dy, rtol=rtol, atol=atol)

      self.assertAllClose(1. / fp8_vars['input_scale_fp8_meta'][0], scale_x)
      self.assertAllClose(1. / fp8_vars['kernel_scale_fp8_meta'][0], scale_k)
      self.assertAllClose(
          1. / fp8_vars['output_grad_scale_fp8_meta'][0], scale_dy)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
