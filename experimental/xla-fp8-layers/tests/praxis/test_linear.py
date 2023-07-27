from functools import partial
from absl.testing import absltest

import numpy as np
import re
import tensorflow as tf

from flax import traverse_util
from fp8layers.praxis import Bias, Linear, FeedForward, MLPBlock
from jax import jit
from jax import numpy as jnp
from jax import random
from jax import value_and_grad
from praxis import base_layer
from praxis import pax_fiddle
from praxis import test_utils
from praxis.layers import activations
from praxis.layers import linears

instantiate = base_layer.instantiate

class LinearsTest(test_utils.TestCase):
  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def testLinearFwd(self):
    di, do = 16, 32

    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)
    inputs = random.uniform(random_key, (48, di)).astype(jnp.bfloat16)

    linear_kwargs = {'input_dims': di, 'output_dims': do,
                     'fprop_dtype': jnp.bfloat16}
    linear_ref: linears.Linear = instantiate(
        pax_fiddle.Config(linears.Linear, name='linear_ref', **linear_kwargs)
    )
    linear_fp8: Linear = instantiate(
        pax_fiddle.Config(Linear, name='linear_fp8', **linear_kwargs)
    )

    vars_ref = linear_ref.init(init_key, inputs)
    vars_fp8 = linear_fp8.init(init_key, inputs)

    def _infer(layer, variables, x):
      y = layer.apply(variables, x)
      return y

    infer_fn_ref = jit(partial(_infer, linear_ref))
    infer_fn_fp8 = jit(partial(_infer, linear_fp8))

    hlo_text = infer_fn_fp8.lower(vars_fp8, inputs).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[48,32]{1,0}', # outputs
            'custom-call',
            'f8e4m3fn[48,16]{1,0}', # inputs
            'f8e4m3fn[32,16]{1,0}', # kernel
            'epilogue',
            'DEFAULT',
        )])),
        msg='output tensor',
    )

    outputs_ref = infer_fn_ref(vars_ref, inputs)
    outputs_fp8 = infer_fn_fp8(vars_fp8, inputs)
    self.assertAllClose(outputs_fp8, outputs_ref, atol=0.1, rtol=0.05)

  def testLinearBwd(self):
    di, do = 16, 32

    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)
    inputs = random.uniform(random_key, (48, di)).astype(jnp.bfloat16)
    dy = random.uniform(random_key, (48, do))

    linear_kwargs = {'input_dims': di, 'output_dims': do,
                     'fprop_dtype': jnp.bfloat16}
    linear_ref: linears.Linear = instantiate(
        pax_fiddle.Config(linears.Linear, name='linear_ref', **linear_kwargs)
    )
    linear_fp8: Linear = instantiate(
        pax_fiddle.Config(Linear, name='linear_fp8', **linear_kwargs)
    )

    vars_ref = linear_ref.init(init_key, inputs)
    vars_fp8 = linear_fp8.init(init_key, inputs)

    def _train(layer, variables, x):
      y = layer.apply(variables, x)
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)

    train_fn_ref = jit(value_and_grad(partial(_train, linear_ref),
                                      argnums=[0, 1]))
    train_fn_fp8 = jit(value_and_grad(partial(_train, linear_fp8),
                                      argnums=[0, 1]))

    hlo_text = train_fn_fp8.lower(vars_fp8, inputs).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[48,32]{1,0}', # outputs
            'custom-call',
            'f8e4m3fn[48,16]{1,0}', # inputs
            'f8e4m3fn[32,16]{1,0}', # kernel
            'epilogue',
            'DEFAULT',
        )])),
        msg='y in forward pass',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[16,32]{1,0}', # dw
            'custom-call',
            'f8e4m3fn[16,48]{1,0}', # inputs
            'f8e5m2[32,48]{1,0}', # dy
            'epilogue',
            'DEFAULT',
        )])),
        msg='dw in backward pass',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[48,16]{1,0}', # dx
            'custom-call',
            'f8e5m2[48,32]{1,0}', # dy
            'f8e4m3fn[16,32]{1,0}', # kernel
            'epilogue',
            'DEFAULT',
        )])),
        msg='dx in backward pass',
    )

    loss_ref, grads_ref = train_fn_ref(vars_ref, inputs)
    loss_fp8, grads_fp8 = train_fn_fp8(vars_fp8, inputs)
    self.assertAllClose(loss_fp8, loss_ref, atol=0.1, rtol=0.05)

    dw_ref, dx_ref = grads_ref[0]['params']['w'], grads_ref[1]
    dw_fp8, dx_fp8 = grads_fp8[0]['params']['w'], grads_fp8[1]
    self.assertAllClose(dw_fp8, dw_ref, atol=0.1, rtol=0.05)
    self.assertAllClose(dx_fp8, dx_ref, atol=0.1, rtol=0.05)

  def testFeedForwardFwd(self):
    di, do = 16, 32

    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    for use_relu in [True, False]:
      for use_bf16 in [True, False]:
        dtype = jnp.bfloat16 if use_bf16 else jnp.float32
        activation = activations.ReLU if use_relu else activations.Identity

        inputs = random.uniform(random_key, (48, di)).astype(dtype)

        ff_kwargs = {'input_dims': di, 'output_dims': do, 'has_bias': True,
                     'bias_init': .3, 'fprop_dtype': dtype,
                     'activation_tpl': pax_fiddle.Config(activation)}
        ff_ref: linears.FeedForward = instantiate(
            pax_fiddle.Config(linears.FeedForward, name='ff_ref', **ff_kwargs)
        )
        ff_fp8: FeedForward = instantiate(
            pax_fiddle.Config(FeedForward, name='ff_fp8', **ff_kwargs)
        )

        vars_ref = ff_ref.init(init_key, inputs)
        vars_fp8 = ff_fp8.init(init_key, inputs)

        def _infer(layer, variables, x):
          y = layer.apply(variables, x)
          return y

        infer_fn_ref = jit(partial(_infer, ff_ref))
        infer_fn_fp8 = jit(partial(_infer, ff_fp8))

        hlo_text = infer_fn_fp8.lower(vars_fp8, inputs).compile().as_text()
        epilog_type = 'BIAS_RELU' if use_relu else 'BIAS'
        out_dtype = 'bf16' if use_bf16 else 'f32'
        self.assertRegex(
            hlo_text,
            re.compile('.*'.join([re.escape(x) for x in (
                'cublas',
                out_dtype + '[48,32]{1,0}', # outputs
                'custom-call',
                'f8e4m3fn[48,16]{1,0}', # inputs
                'f8e4m3fn[32,16]{1,0}', # kernel
                'bf16[32]{0}', # bias
                'epilogue',
                epilog_type,
            )])),
            msg='output tensor',
        )

        outputs_ref = infer_fn_ref(vars_ref, inputs)
        outputs_fp8 = infer_fn_fp8(vars_fp8, inputs)
        self.assertAllClose(outputs_fp8, outputs_ref, atol=0.1, rtol=0.05)

  def testFeedForwardBwd(self):
    di, do = 16, 32

    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    for use_bf16 in [True, False]:
      dtype = jnp.bfloat16 if use_bf16 else jnp.float32

      inputs = random.uniform(random_key, (48, di)).astype(dtype)
      dy = random.uniform(random_key, (48, do))

      # TODO(kaixih): It seems the ReLU won't be fused in the fwd+bwd case. It
      # may be due to some of the intermediate results will be consumed by the
      # bprop. Double-check if it is the case.
      ff_kwargs = {'input_dims': di, 'output_dims': do, 'has_bias': True,
                   'bias_init': .3, 'fprop_dtype': dtype,
                   'activation_tpl': pax_fiddle.Config(activations.Identity)}
      ff_ref: linears.FeedForward = instantiate(
          pax_fiddle.Config(linears.FeedForward, name='ff_ref', **ff_kwargs)
      )
      ff_fp8: FeedForward = instantiate(
          pax_fiddle.Config(FeedForward, name='ff_fp8', **ff_kwargs)
      )

      vars_ref = ff_ref.init(init_key, inputs)
      vars_fp8 = ff_fp8.init(init_key, inputs)

      def _train(layer, variables, x):
        y = layer.apply(variables, x)
        loss = y * dy.astype(y.dtype)
        return jnp.mean(loss)

      train_fn_ref = jit(value_and_grad(partial(_train, ff_ref),
                                        argnums=[0, 1]))
      train_fn_fp8 = jit(value_and_grad(partial(_train, ff_fp8),
                                        argnums=[0, 1]))

      hlo_text = train_fn_fp8.lower(vars_fp8, inputs).compile().as_text()
      out_dtype = 'bf16' if use_bf16 else 'f32'
      self.assertRegex(
          hlo_text,
          re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              out_dtype + '[48,32]{1,0}', # outputs
              'custom-call',
              'f8e4m3fn[48,16]{1,0}', # inputs
              'f8e4m3fn[32,16]{1,0}', # kernel
              'bf16[32]{0}', # bias
              'epilogue',
              'BIAS',
          )])),
          msg='y in forward pass',
      )
      self.assertRegex(
          hlo_text,
          re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              out_dtype + '[16,32]{1,0}', # dw
              'custom-call',
              'f8e4m3fn[16,48]{1,0}', # inputs
              'f8e5m2[32,48]{1,0}', # dy
              'epilogue',
              'DEFAULT',
          )])),
          msg='dw in backward pass',
      )
      self.assertRegex(
          hlo_text,
          re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              out_dtype + '[48,16]{1,0}', # dx
              'custom-call',
              'f8e5m2[48,32]{1,0}', # dy
              'f8e4m3fn[16,32]{1,0}', # kernel
              'epilogue',
              'DEFAULT',
          )])),
          msg='dx in backward pass',
      )

      loss_ref, grads_ref = train_fn_ref(vars_ref, inputs)
      loss_fp8, grads_fp8 = train_fn_fp8(vars_fp8, inputs)
      self.assertAllClose(loss_fp8, loss_ref, atol=0.1, rtol=0.05)

      flat_ref = traverse_util.flatten_dict(grads_ref[0]['params'], sep='/')
      flat_fp8 = traverse_util.flatten_dict(grads_fp8[0]['params'], sep='/')

      w_names = ['linear/w', 'bias/b']
      for w_name in w_names:
        dw_ref, dw_fp8 = flat_ref[w_name], flat_fp8[w_name]
        self.assertAllClose(dw_fp8, dw_ref, atol=0.1, rtol=0.05)

      dx_ref, dx_fp8 = grads_ref[1], grads_fp8[1]
      self.assertAllClose(dx_fp8, dx_ref, atol=0.1, rtol=0.05)

  def testMLPBlockFwd(self):
    di, do = 16, 32
    dtype = jnp.bfloat16

    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)
    inputs = random.uniform(random_key, (48, di)).astype(dtype)

    ff_kwargs = {'input_dims': di, 'output_dims': do, 'has_bias': True,
                 'bias_init': .3, 'fprop_dtype': dtype,
                 'activation_tpl': pax_fiddle.Config(activations.ReLU)}
    ff_ref = pax_fiddle.Config(linears.FeedForward, name='ff_ref', **ff_kwargs)
    ff_fp8 = pax_fiddle.Config(FeedForward, name='ff_fp8', **ff_kwargs)

    mlp_kwargs = {'num_layers': 2, 'hidden_dims': 64, 'dtype': dtype}
    mlp_ref: linears.MLPBlock = instantiate(
        pax_fiddle.Config(linears.MLPBlock, name='mlp_ref', ff_tpl=ff_ref,
                          **mlp_kwargs)
    )
    mlp_fp8: MLPBlock = instantiate(
        pax_fiddle.Config(MLPBlock, name='mlp_fp8', ff_tpl=ff_fp8, **mlp_kwargs)
    )

    vars_ref = mlp_ref.init(init_key, inputs)
    vars_fp8 = mlp_fp8.init(init_key, inputs)

    def _infer(layer, variables, x):
      y = layer.apply(variables, x)
      return y

    infer_fn_ref = jit(partial(_infer, mlp_ref))
    infer_fn_fp8 = jit(partial(_infer, mlp_fp8))

    hlo_text = infer_fn_fp8.lower(vars_fp8, inputs).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f8e4m3fn[48,64]{1,0}', # outputs
            'custom-call',
            'f8e4m3fn[48,16]{1,0}', # inputs
            'f8e4m3fn[64,16]{1,0}', # kernel
            'bf16[64]{0}', # bias
            'epilogue',
            'BIAS_RELU',
        )])),
        msg='1st output tensor',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[48,32]{1,0}', # outputs
            'custom-call',
            'f8e4m3fn[48,64]{1,0}', # inputs
            'f8e4m3fn[32,64]{1,0}', # kernel
            'bf16[32]{0}', # bias
            'epilogue',
            'BIAS_RELU',
        )])),
        msg='2nd output tensor',
    )

    outputs_ref = infer_fn_ref(vars_ref, inputs)
    outputs_fp8 = infer_fn_fp8(vars_fp8, inputs)
    self.assertAllClose(outputs_fp8, outputs_ref, atol=0.1, rtol=0.05)

  def testMLPBlockBwd(self):
    di, do = 16, 32
    dtype = jnp.bfloat16

    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)
    inputs = random.uniform(random_key, (48, di)).astype(dtype)
    dy = random.uniform(random_key, (48, do))

    ff_kwargs = {'input_dims': di, 'output_dims': do, 'has_bias': True,
                 'bias_init': .3, 'fprop_dtype': dtype,
                 'activation_tpl': pax_fiddle.Config(activations.ReLU)}
    ff_ref = pax_fiddle.Config(linears.FeedForward, name='ff_ref', **ff_kwargs)
    ff_fp8 = pax_fiddle.Config(FeedForward, name='ff_fp8', **ff_kwargs)

    mlp_kwargs = {'num_layers': 2, 'hidden_dims': 64, 'dtype': dtype}
    mlp_ref: linears.MLPBlock = instantiate(
        pax_fiddle.Config(linears.MLPBlock, name='mlp_ref', ff_tpl=ff_ref,
                          **mlp_kwargs)
    )
    mlp_fp8: MLPBlock = instantiate(
        pax_fiddle.Config(MLPBlock, name='mlp_fp8', ff_tpl=ff_fp8, **mlp_kwargs)
    )

    vars_ref = mlp_ref.init(init_key, inputs)
    vars_fp8 = mlp_fp8.init(init_key, inputs)

    def _train(layer, variables, x):
      y = layer.apply(variables, x)
      loss = y * dy.astype(y.dtype)
      return jnp.mean(loss)

    train_fn_ref = jit(value_and_grad(partial(_train, mlp_ref), argnums=[0, 1]))
    train_fn_fp8 = jit(value_and_grad(partial(_train, mlp_fp8), argnums=[0, 1]))

    hlo_text = train_fn_fp8.lower(vars_fp8, inputs).compile().as_text()
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[48,64]{1,0}', # outputs
            'custom-call',
            'f8e4m3fn[48,16]{1,0}', # inputs
            'f8e4m3fn[64,16]{1,0}', # kernel
            'bf16[64]{0}', # bias
            'epilogue',
            'BIAS',
        )])),
        msg='1st output in forward pass',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[48,32]{1,0}', # outputs
            'custom-call',
            'f8e4m3fn[48,64]{1,0}', # inputs
            'f8e4m3fn[32,64]{1,0}', # kernel
            'bf16[32]{0}', # bias
            'epilogue',
            'BIAS',
        )])),
        msg='2nd output in forward pass',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[64,32]{1,0}', # dw
            'custom-call',
            'f8e4m3fn[64,48]{1,0}', # inputs
            'f8e5m2[32,48]{1,0}', # dy
            'epilogue',
            'DEFAULT',
        )])),
        msg='2nd dw in backward pass',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[48,64]{1,0}', # dx
            'custom-call',
            'f8e5m2[48,32]{1,0}', # dy
            'f8e4m3fn[64,32]{1,0}', # kernel
            'epilogue',
            'DEFAULT',
        )])),
        msg='2nd dx in backward pass',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[16,64]{1,0}', # dw
            'custom-call',
            'f8e4m3fn[16,48]{1,0}', # inputs
            'f8e5m2[64,48]{1,0}', # dy
            'epilogue',
            'DEFAULT',
        )])),
        msg='1st dw in backward pass',
    )
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'bf16[48,16]{1,0}', # dx
            'custom-call',
            'f8e5m2[48,64]{1,0}', # dy
            'f8e4m3fn[16,64]{1,0}', # kernel
            'epilogue',
            'DEFAULT',
        )])),
        msg='1st dx in backward pass',
    )

    loss_ref, grads_ref = train_fn_ref(vars_ref, inputs)
    loss_fp8, grads_fp8 = train_fn_fp8(vars_fp8, inputs)
    self.assertAllClose(loss_fp8, loss_ref, atol=0.1, rtol=0.05)

    flat_ref = traverse_util.flatten_dict(grads_ref[0]['params'], sep='/')
    flat_fp8 = traverse_util.flatten_dict(grads_fp8[0]['params'], sep='/')

    w_names = ['mlp_layers_0/linear/w', 'mlp_layers_0/bias/b',
               'mlp_layers_1/linear/w', 'mlp_layers_1/bias/b']
    for w_name in w_names:
      dw_ref, dw_fp8 = flat_ref[w_name], flat_fp8[w_name]
      self.assertAllClose(dw_fp8, dw_ref, atol=0.1, rtol=0.05)

    dx_ref, dx_fp8 = grads_ref[1], grads_fp8[1]
    self.assertAllClose(dx_fp8, dx_ref, atol=0.1, rtol=0.05)


if __name__ == '__main__':
  absltest.main()

