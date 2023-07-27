"""Tests for the fp8 layers with partitioning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import re

import optax

import jax._src.test_util as jtu
from jax import device_count
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import value_and_grad

from fp8layers.flax import DenseGeneral

# Sharding related
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils
from flax.linen import spmd # Flax Linen SPMD.

def get_hlo_text(rules):
  device_mesh = mesh_utils.create_device_mesh((4, 2))
  mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
  
  model = DenseGeneral(8192, use_bias=False, kernel_axes=('hidden', 'mlp'))
  
  x = random.normal(random.PRNGKey(0), (8192, 8192))
  dy = random.normal(random.PRNGKey(0), (8192, 8192))
  k = random.PRNGKey(0)
  
  spmd.set_logical_axis_rules(rules)
  
  initialized_state = model.init(k, x)
  
  def loss_fn(state, x, dy):
    x = spmd.with_logical_constraint(x, ('batch', 'embed'))
    dy = spmd.with_logical_constraint(dy, ('batch', 'mlp'))

    y = model.apply(state, x)
    loss = y * dy.astype(y.dtype)
    return jnp.sum(loss)
  
  pjit_step_fn = pjit(
      value_and_grad(loss_fn, argnums=[0]),
  )
  
  with mesh:
    lowered = pjit_step_fn.lower(initialized_state, x, dy)
  hlo = lowered.compile()
  return hlo.as_text()


@jtu.with_config(jax_numpy_rank_promotion='allow',
                 jax_numpy_dtype_promotion='standard')
class PartitionTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

  def testAllReduceDp(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    rules = (('batch', 'data'),
             ('fp8_meta', None))

    hlo_text = get_hlo_text(rules)
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[2048,8192]{1,0}',
            'custom-call',
            'f8e4m3fn[2048,8192]{1,0}',
            'f8e4m3fn[8192,8192]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='y tensor')
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[8192,8192]{1,0}',
            'custom-call',
            'f8e4m3fn[8192,2048]{1,0}',
            'f8e5m2[8192,2048]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='dw tensor')

    # The all-reduce for the input and output_grads follows the same replica
    # group. So, it accepts two operands and return a tuple.
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'all-reduce',
            '(f32[], f32[])', # output
            'all-reduce',
            'f32[]', # input
            'f32[]', # input
            'replica_groups={{0,2,4,6},{1,3,5,7}}',
        )])), msg="all-reduce on input and output_grads amax")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[8192,8192]', # output
              'all-reduce',
              'f32[8192,8192]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on intermediate results of dy*x=dk matmul")


  def testAllReduceTpRow(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    rules = (('hidden', 'model'),)

    hlo_text = get_hlo_text(rules)

    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[8192,8192]{1,0}',
            'custom-call',
            'f8e4m3fn[8192,4096]{1,0}',
            'f8e4m3fn[8192,4096]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='y tensor')
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[4096,8192]{1,0}',
            'custom-call',
            'f8e4m3fn[4096,8192]{1,0}',
            'f8e5m2[8192,8192]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='dw tensor')

    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'all-reduce',
            'f32[]', # output
            'all-reduce',
            'f32[]', # input
            'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
        )])), msg="all-reduce on kernel amax")
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[8192,8192]', # output
              'all-reduce',
              'f32[8192,8192]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on intermediate results of x*k=y matmul")
  

  def testAllReduceTpCol(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    rules = (('mlp', 'model'),)

    hlo_text = get_hlo_text(rules)

    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[8192,4096]{1,0}',
            'custom-call',
            'f8e4m3fn[8192,8192]{1,0}',
            'f8e4m3fn[4096,8192]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='y tensor')
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[8192,4096]{1,0}',
            'custom-call',
            'f8e4m3fn[8192,8192]{1,0}',
            'f8e5m2[4096,8192]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='dw tensor')

   # The all-reduce for the kernel and output_grads follows the same replica
    # group. So, it accepts two operands and return a tuple.
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'all-reduce',
            '(f32[], f32[])', # output
            'all-reduce',
            'f32[]', # input
            'f32[]', # input
            'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
        )])), msg="all-reduce on kernel and output_grad amax")


  def testAllReduceDpTpRow(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    rules = (('batch', 'data'),
             ('hidden', 'model'))

    hlo_text = get_hlo_text(rules)
    
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[2048,8192]{1,0}',
            'custom-call',
            'f8e4m3fn[2048,4096]{1,0}',
            'f8e4m3fn[8192,4096]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='y tensor')
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[4096,8192]{1,0}',
            'custom-call',
            'f8e4m3fn[4096,2048]{1,0}',
            'f8e5m2[8192,2048]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='dw tensor')

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on kernel amax")

    # The all-reduce for the kernel and output_grads follows the same replica
    # group. So, it accepts two operands and return a tuple.
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              '(f32[], f32[])', # output
              'all-reduce',
              'f32[]', # input
              'f32[]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on input and output_grads amax")
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[2048,8192]', # output
              'all-reduce',
              'f32[2048,8192]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on intermediate results of x*k=y matmul")
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[4096,8192]', # output
              'all-reduce',
              'f32[4096,8192]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on intermediate results of dy*x=dk matmul")


  def testAllReduceDpTpCol(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    rules = (('batch', 'data'),
             ('mlp', 'model'))

    hlo_text = get_hlo_text(rules)

    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[2048,4096]{1,0}',
            'custom-call',
            'f8e4m3fn[2048,8192]{1,0}',
            'f8e4m3fn[4096,8192]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='y tensor')
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[8192,4096]{1,0}',
            'custom-call',
            'f8e4m3fn[8192,2048]{1,0}',
            'f8e5m2[4096,2048]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='dw tensor')

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on kernel amax")
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on input amax")

    # The loss function applies sum on the output, which will also generate an
    # all-reduce over the same replica groups as the output grad amax.
    count = len(re.findall(
        re.compile('.*'.join([re.escape(x) for x in (
            'all-reduce',
            'f32[]', # output
            'all-reduce',
            'f32[]', # input
            'replica_groups={{0,1,2,3,4,5,6,7}}',
        )])), hlo_text))
    self.assertEqual(2, count, msg="all-reduce on output_grad amax and loss")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[8192,4096]', # output
              'all-reduce',
              'f32[8192,4096]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on intermediate results of dy*x=dk matmul")


  def testAllReduceOptimizedDpTpRow(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    rules = (('batch', 'data'),
             ('embed', 'model'),
             ('hidden', 'model'))

    hlo_text = get_hlo_text(rules)

    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[2048,8192]{1,0}',
            'custom-call',
            'f8e4m3fn[2048,4096]{1,0}',
            'f8e4m3fn[8192,4096]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='y tensor')
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'cublas',
            'f32[4096,8192]{1,0}',
            'custom-call',
            'f8e4m3fn[4096,2048]{1,0}',
            'f8e5m2[8192,2048]{1,0}',
            'epilogue',
            'DEFAULT',
        )])), msg='dw tensor')

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on kernel amax")
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,1,2,3,4,5,6,7}}',
          )])), msg="all-reduce on input amax")

    # The loss function applies sum on the output, which will also generate an
    # all-reduce over the same replica groups as the output grad amax.
    count = len(re.findall(
        re.compile('.*'.join([re.escape(x) for x in (
            'all-reduce',
            'f32[]', # output
            'all-reduce',
            'f32[]', # input
            'replica_groups={{0,2,4,6},{1,3,5,7}}',
        )])), hlo_text))
    self.assertEqual(2, count, msg="all-reduce on output_grad amax and loss")
    
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[2048,8192]', # output
              'all-reduce',
              'f32[2048,8192]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on intermediate results of x*k=y matmul")
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[4096,8192]', # output
              'all-reduce',
              'f32[4096,8192]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on intermediate results of dy*x=dk matmul")


  def testAllReduceFullSharding(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    rules = (('batch', 'data'),
             ('embed', 'model'),
             ('hidden', 'data'),
             ('mlp', 'model'))

    hlo_text = get_hlo_text(rules)
    # TODO(shuw): Add type check when all-gather is included in whitelist.
    # The all-reduce for the amax follows the same replica group.
    self.assertRegex(
        hlo_text, re.compile('.*'.join([re.escape(x) for x in (
            'all-reduce',
            '(f32[], f32[], f32[])', # output
            'all-reduce',
            'f32[]', # input
            'f32[]', # input
            'f32[]', # input
            'replica_groups={{0,1,2,3,4,5,6,7}}',
        )])), msg="all-reduce on input, kernel, and output_grads amax")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-gather',
              'f32[2048,8192]', # output
              'all-gather',
              'f32[2048,4096]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-gather on input x for matmul")
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-gather',
              'f32[8192,4096]', # output
              'all-gather',
              'f32[2048,4096]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-gather on kernel k for matmul")


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
