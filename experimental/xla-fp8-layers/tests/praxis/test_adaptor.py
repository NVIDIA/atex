from functools import partial
from absl.testing import absltest

import numpy as np
import re
import tensorflow as tf

from flax import traverse_util
from jax import jit
from jax import numpy as jnp
from jax import random
from jax import value_and_grad
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import activations
from praxis.layers import linears

from fp8layers.flax import fp8_projection
from fp8layers.flax import fp8_qkv_projection
from fp8layers.flax import fp8_qkv_combined_projection
from fp8layers.flax import fp8_attention_output_projection

instantiate = base_layer.instantiate

def projection(inputs, w, use_bias, b):
  """Reference code to perform typical projection and bias addition."""
  eqn = '...K,KN->...N'
  ret = jnp.einsum(eqn, inputs, w)
  if use_bias:
    ret += b
  return ret

def qkv_combined_projection(inputs, w, use_bias, b):
  """Reference code to perform qkv combined projection and bias addition."""
  rank = len(inputs.shape)
  batch_dims_rank = rank - 1
  # K indexes qkv.
  eqn = '...D,KDNH->K...NH'
  ret = jnp.einsum(eqn, inputs, w)

  if use_bias:
    # Add newaxis to bias weight for each batch dim since ret is K...NH
    # and theta.b is KNH. Need to reshape theta.b to K...NH
    ret += jnp.expand_dims(b, list(range(1, batch_dims_rank + 1)))
  return ret

def qkv_projection(inputs, w, use_bias, b):
  """Reference code to perform qkv projection and bias addition."""
  eqn = '...D,DNH->...NH'
  ret = jnp.einsum(eqn, inputs, w)
  if use_bias:
    ret += b
  return ret

def attention_output_projection(inputs, w, use_bias, b, use_nhd_shape):
  """Reference code to perform attn out projection and bias addition."""
  if use_nhd_shape:
    eqn = '...NH,NHD->...D'
  else:
    eqn = '...NH,DNH->...D'
  ret = jnp.einsum(eqn, inputs, w)
  if use_bias:
    ret += b
  return ret


class EinsumToDotAdaptorTest(test_utils.TestCase):
  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def testQKVCombinedProjFwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, D, N, H, K = (12, 48, 64, 16, 32, 3)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, D)).astype(dtype)
    w = random.uniform(random_key, (K, D, N, H)).astype(dtype)
    b = random.uniform(random_key, (K, N, H)).astype(dtype)

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _infer_ref(x, var, use_bias):
      y = qkv_combined_projection(x, var['w'], use_bias, var['b'])
      return y

    def _infer_fp8(x, var, use_bias):
      y = fp8_qkv_combined_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'])
      return y

    infer_fn_ref = jit(partial(_infer_ref, use_bias=True))
    infer_fn_fp8 = jit(partial(_infer_fp8, use_bias=True))

    var_ref = {'w': w, 'b': b}
    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}
    hlo_text = infer_fn_fp8.lower(x, var_fp8).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{K*N*H}]{{1,0}}', # outputs
            'custom-call',
            f'f8e4m3fn[{B*T},{D}]{{1,0}}', # inputs
            f'f8e4m3fn[{K*N*H},{D}]{{1,0}}', # kernel
            'epilogue',
            'BIAS',
        )])),
        msg='output tensor',
    )

    y_ref = infer_fn_ref(x, var_ref)
    y_fp8 = infer_fn_fp8(x, var_fp8)

    self.assertAllClose(y_fp8, y_ref, atol=0.1, rtol=0.05)

  def testQKVCombinedProjBwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, D, N, H, K = (12, 48, 64, 16, 32, 3)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, D)).astype(dtype)
    w = random.uniform(random_key, (K, D, N, H)).astype(dtype)
    b = random.uniform(random_key, (K, N, H)).astype(dtype)
    dy = random.uniform(random_key, (K, B, T, N, H))

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _train_ref(x, var, use_bias):
      y = qkv_combined_projection(x, var['w'], use_bias, var['b'])
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)

    def _train_fp8(x, var, use_bias):
      y = fp8_qkv_combined_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'])
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)


    train_fn_ref = jit(
        value_and_grad(partial(_train_ref, use_bias=True), argnums=[0, 1]))
    train_fn_fp8 = jit(
        value_and_grad(partial(_train_fp8, use_bias=True), argnums=[0, 1]))

    var_ref = {'w': w, 'b': b}
    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}
    hlo_text = train_fn_fp8.lower(x, var_fp8).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{K*N*H}]{{1,0}}', # outputs
            'custom-call',
            f'f8e4m3fn[{B*T},{D}]{{1,0}}', # inputs
            f'f8e4m3fn[{K*N*H},{D}]{{1,0}}', # kernel
            'epilogue',
            'BIAS',
        )])),
        msg='y tensor',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{D}]{{1,0}}', # dx
            'custom-call',
            f'f8e5m2[{B*T},{K*N*H}]{{1,0}}', # dy
            f'f8e4m3fn[{D},{K*N*H}]{{1,0}}', # w
            'epilogue',
            'DEFAULT',
        )])),
        msg='dx tensor',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{D},{K*N*H}]{{1,0}}', # dw
            'custom-call',
            f'f8e4m3fn[{D},{B*T}]{{1,0}}', # x
            f'f8e5m2[{K*N*H},{B*T}]{{1,0}}', # dy
            'epilogue',
            'DEFAULT',
        )])),
        msg='dw tensor',
    )

    loss_ref, grads_ref = train_fn_ref(x, var_ref)
    loss_fp8, grads_fp8 = train_fn_fp8(x, var_fp8)
    self.assertAllClose(loss_fp8, loss_ref, atol=0.1, rtol=0.05)

    dw_ref, dx_ref = grads_ref[1]['w'], grads_ref[0]
    dw_fp8, dx_fp8 = grads_fp8[1]['w'], grads_fp8[0]
    self.assertAllClose(dw_fp8, dw_ref, atol=0.1, rtol=0.05)
    self.assertAllClose(dx_fp8, dx_ref, atol=0.1, rtol=0.05)

  def testAttnOutProjFwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, D, N, H = (12, 48, 64, 16, 32)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, N, H)).astype(dtype)
    w = random.uniform(random_key, (N, H, D)).astype(dtype)
    b = random.uniform(random_key, (D,)).astype(dtype)

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _infer_ref(x, var, use_bias, use_nhd_shape):
      y = attention_output_projection(x, var['w'], use_bias, var['b'],
                                      use_nhd_shape)
      return y

    def _infer_fp8(x, var, use_bias, use_nhd_shape):
      y = fp8_attention_output_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'], use_nhd_shape)
      return y

    infer_fn_ref = jit(partial(_infer_ref, use_bias=True, use_nhd_shape=True))
    infer_fn_fp8 = jit(partial(_infer_fp8, use_bias=True, use_nhd_shape=True))

    var_ref = {'w': w, 'b': b}
    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}
    hlo_text = infer_fn_fp8.lower(x, var_fp8).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{D}]{{1,0}}', # outputs
            'custom-call',
            f'f8e4m3fn[{B*T},{N*H}]{{1,0}}', # inputs
            f'f8e4m3fn[{D},{N*H}]{{1,0}}', # kernel
            'epilogue',
            'BIAS',
        )])),
        msg='output tensor',
    )

    y_ref = infer_fn_ref(x, var_ref)
    y_fp8 = infer_fn_fp8(x, var_fp8)

    self.assertAllClose(y_fp8, y_ref, atol=0.1, rtol=0.05)

  def testAttnOutProjBwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, D, N, H = (12, 48, 64, 16, 32)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, N, H)).astype(dtype)
    w = random.uniform(random_key, (N, H, D)).astype(dtype)
    b = random.uniform(random_key, (D,)).astype(dtype)
    dy = random.uniform(random_key, (B, T, D))

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _train_ref(x, var, use_bias, use_nhd_shape):
      y = attention_output_projection(x, var['w'], use_bias, var['b'],
                                      use_nhd_shape)
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)

    def _train_fp8(x, var, use_bias, use_nhd_shape):
      y = fp8_attention_output_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'], use_nhd_shape)
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)


    train_fn_ref = jit(
        value_and_grad(partial(_train_ref, use_bias=True, use_nhd_shape=True),
                       argnums=[0, 1]))
    train_fn_fp8 = jit(
        value_and_grad(partial(_train_fp8, use_bias=True, use_nhd_shape=True),
                       argnums=[0, 1]))

    var_ref = {'w': w, 'b': b}
    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}
    hlo_text = train_fn_fp8.lower(x, var_fp8).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{D}]{{1,0}}', # outputs
            'custom-call',
            f'f8e4m3fn[{B*T},{N*H}]{{1,0}}', # inputs
            f'f8e4m3fn[{D},{N*H}]{{1,0}}', # kernel
            'epilogue',
            'BIAS',
        )])),
        msg='y tensor',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{N*H}]{{1,0}}', # dx
            'custom-call',
            f'f8e5m2[{B*T},{D}]{{1,0}}', # dy
            f'f8e4m3fn[{N*H},{D}]{{1,0}}', # w
            'epilogue',
            'DEFAULT',
        )])),
        msg='dx tensor',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{N*H},{D}]{{1,0}}', # dw
            'custom-call',
            f'f8e4m3fn[{N*H},{B*T}]{{1,0}}', # x
            f'f8e5m2[{D},{B*T}]{{1,0}}', # dy
            'epilogue',
            'DEFAULT',
        )])),
        msg='dw tensor',
    )

    loss_ref, grads_ref = train_fn_ref(x, var_ref)
    loss_fp8, grads_fp8 = train_fn_fp8(x, var_fp8)
    self.assertAllClose(loss_fp8, loss_ref, atol=0.1, rtol=0.05)

    dw_ref, dx_ref = grads_ref[1]['w'], grads_ref[0]
    dw_fp8, dx_fp8 = grads_fp8[1]['w'], grads_fp8[0]
    self.assertAllClose(dw_fp8, dw_ref, atol=0.1, rtol=0.05)
    self.assertAllClose(dx_fp8, dx_ref, atol=0.1, rtol=0.05)


  def testQKVProjFwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, D, N, H = (12, 48, 64, 16, 32)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, D)).astype(dtype)
    w = random.uniform(random_key, (D, N, H)).astype(dtype)
    b = random.uniform(random_key, (N, H)).astype(dtype)

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _infer_ref(x, var, use_bias):
      y = qkv_projection(x, var['w'], use_bias, var['b'])
      return y

    def _infer_fp8(x, var, use_bias):
      y = fp8_qkv_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'])
      return y

    infer_fn_ref = jit(partial(_infer_ref, use_bias=True))
    infer_fn_fp8 = jit(partial(_infer_fp8, use_bias=True))

    var_ref = {'w': w, 'b': b}
    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}
    hlo_text = infer_fn_fp8.lower(x, var_fp8).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{N*H}]{{1,0}}', # outputs
            'custom-call',
            f'f8e4m3fn[{B*T},{D}]{{1,0}}', # inputs
            f'f8e4m3fn[{N*H},{D}]{{1,0}}', # kernel
            'epilogue',
            'BIAS',
        )])),
        msg='output tensor',
    )

    y_ref = infer_fn_ref(x, var_ref)
    y_fp8 = infer_fn_fp8(x, var_fp8)

    self.assertAllClose(y_fp8, y_ref, atol=0.1, rtol=0.05)

  def testQKVProjBwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, D, N, H = (12, 48, 64, 16, 32)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, D)).astype(dtype)
    w = random.uniform(random_key, (D, N, H)).astype(dtype)
    b = random.uniform(random_key, (N, H)).astype(dtype)
    dy = random.uniform(random_key, (B, T, N, H))

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _train_ref(x, var, use_bias):
      y = qkv_projection(x, var['w'], use_bias, var['b'])
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)

    def _train_fp8(x, var, use_bias):
      y = fp8_qkv_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'])
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)

    train_fn_ref = jit(
        value_and_grad(partial(_train_ref, use_bias=True), argnums=[0, 1]))
    train_fn_fp8 = jit(
        value_and_grad(partial(_train_fp8, use_bias=True), argnums=[0, 1]))

    var_ref = {'w': w, 'b': b}
    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}
    hlo_text = train_fn_fp8.lower(x, var_fp8).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{N*H}]{{1,0}}', # outputs
            'custom-call',
            f'f8e4m3fn[{B*T},{D}]{{1,0}}', # inputs
            f'f8e4m3fn[{N*H},{D}]{{1,0}}', # kernel
            'epilogue',
            'BIAS',
        )])),
        msg='y tensor',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{D}]{{1,0}}', # dx
            'custom-call',
            f'f8e5m2[{B*T},{N*H}]{{1,0}}', # dy
            f'f8e4m3fn[{D},{N*H}]{{1,0}}', # w
            'epilogue',
            'DEFAULT',
        )])),
        msg='dx tensor',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{D},{N*H}]{{1,0}}', # dw
            'custom-call',
            f'f8e4m3fn[{D},{B*T}]{{1,0}}', # x
            f'f8e5m2[{N*H},{B*T}]{{1,0}}', # dy
            'epilogue',
            'DEFAULT',
        )])),
        msg='dw tensor',
    )

    loss_ref, grads_ref = train_fn_ref(x, var_ref)
    loss_fp8, grads_fp8 = train_fn_fp8(x, var_fp8)
    self.assertAllClose(loss_fp8, loss_ref, atol=0.1, rtol=0.05)

    dw_ref, dx_ref = grads_ref[1]['w'], grads_ref[0]
    dw_fp8, dx_fp8 = grads_fp8[1]['w'], grads_fp8[0]
    self.assertAllClose(dw_fp8, dw_ref, atol=0.1, rtol=0.05)
    self.assertAllClose(dx_fp8, dx_ref, atol=0.1, rtol=0.05)

  def testProjFwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, K, N = (12, 48, 64, 16)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, K)).astype(dtype)
    w = random.uniform(random_key, (K, N)).astype(dtype)
    b = random.uniform(random_key, (N,)).astype(dtype)

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _infer_ref(x, var, use_bias):
      y = projection(x, var['w'], use_bias, var['b'])
      return y

    def _infer_fp8(x, var, use_bias):
      y = fp8_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'])
      return y

    infer_fn_ref = jit(partial(_infer_ref, use_bias=True))
    infer_fn_fp8 = jit(partial(_infer_fp8, use_bias=True))

    var_ref = {'w': w, 'b': b}
    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}
    hlo_text = infer_fn_fp8.lower(x, var_fp8).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{N}]{{1,0}}', # outputs
            'custom-call',
            f'f8e4m3fn[{B*T},{K}]{{1,0}}', # inputs
            f'f8e4m3fn[{N},{K}]{{1,0}}', # kernel
            'epilogue',
            'BIAS',
        )])),
        msg='output tensor',
    )

    y_ref = infer_fn_ref(x, var_ref)
    y_fp8 = infer_fn_fp8(x, var_fp8)

    self.assertAllClose(y_fp8, y_ref, atol=0.1, rtol=0.05)

  def testProjBwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, K, N = (12, 48, 64, 16)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, K)).astype(dtype)
    w = random.uniform(random_key, (K, N)).astype(dtype)
    b = random.uniform(random_key, (N,)).astype(dtype)
    dy = random.uniform(random_key, (B, T, N))

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _train_ref(x, var, use_bias):
      y = projection(x, var['w'], use_bias, var['b'])
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)

    def _train_fp8(x, var, use_bias):
      y = fp8_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'])
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)

    train_fn_ref = jit(
        value_and_grad(partial(_train_ref, use_bias=True), argnums=[0, 1]))
    train_fn_fp8 = jit(
        value_and_grad(partial(_train_fp8, use_bias=True), argnums=[0, 1]))

    var_ref = {'w': w, 'b': b}
    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}
    hlo_text = train_fn_fp8.lower(x, var_fp8).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{N}]{{1,0}}', # outputs
            'custom-call',
            f'f8e4m3fn[{B*T},{K}]{{1,0}}', # inputs
            f'f8e4m3fn[{N},{K}]{{1,0}}', # kernel
            'epilogue',
            'BIAS',
        )])),
        msg='y tensor',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{B*T},{K}]{{1,0}}', # dx
            'custom-call',
            f'f8e5m2[{B*T},{N}]{{1,0}}', # dy
            f'f8e4m3fn[{K},{N}]{{1,0}}', # w
            'epilogue',
            'DEFAULT',
        )])),
        msg='dx tensor',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            f'bf16[{K},{N}]{{1,0}}', # dw
            'custom-call',
            f'f8e4m3fn[{K},{B*T}]{{1,0}}', # x
            f'f8e5m2[{N},{B*T}]{{1,0}}', # dy
            'epilogue',
            'DEFAULT',
        )])),
        msg='dw tensor',
    )

    loss_ref, grads_ref = train_fn_ref(x, var_ref)
    loss_fp8, grads_fp8 = train_fn_fp8(x, var_fp8)
    self.assertAllClose(loss_fp8, loss_ref, atol=0.1, rtol=0.05)

    dw_ref, dx_ref = grads_ref[1]['w'], grads_ref[0]
    dw_fp8, dx_fp8 = grads_fp8[1]['w'], grads_fp8[0]
    self.assertAllClose(dw_fp8, dw_ref, atol=0.1, rtol=0.05)
    self.assertAllClose(dx_fp8, dx_ref, atol=0.1, rtol=0.05)

if __name__ == '__main__':
  absltest.main()
