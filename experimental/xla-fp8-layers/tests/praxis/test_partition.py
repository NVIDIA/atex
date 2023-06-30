"""Tests for the fp8 layers with partitioning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import re

import jax._src.test_util as jtu
from jax import device_count
from jax import numpy as jnp
from jax import random
from jax import value_and_grad

# Sharding related
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from flax.linen import spmd # Flax Linen SPMD.

from fp8layers.praxis import Linear

from praxis import base_layer
from praxis import pax_fiddle

WeightInit = base_layer.WeightInit
instantiate = base_layer.instantiate


def get_hlo_text(rules):
  mesh_names = ('data', 'model')
  mesh_shape = (4, 2)
  device_mesh = mesh_utils.create_device_mesh(mesh_shape)
  mesh = Mesh(devices=device_mesh, axis_names=mesh_names)

  linear_kwargs = {'input_dims': 8192, 'output_dims': 8192,
                   'kernel_axes': ('hidden', 'mlp'),
                   'logical_axes_rules': rules,
                   'mesh_axis_names': mesh_names,
                   'ici_mesh_shape': mesh_shape}
  model: Linear = instantiate(
      pax_fiddle.Config(Linear, name='linear_fp8', **linear_kwargs)
  )
  
  x = random.normal(random.PRNGKey(0), (8192, 8192))
  dy = random.normal(random.PRNGKey(0), (8192, 8192))
  k = random.PRNGKey(0)
  
  spmd.set_logical_axis_rules(rules)
  
  initialized_state = model.init(k, x, mutable=['params', 'fp8_params'])

  def loss_fn(state, x, dy):
    x = spmd.with_logical_constraint(x, ('batch', 'embed'))
    dy = spmd.with_logical_constraint(dy, ('batch', 'mlp'))

    y = model.apply(state, x)
    loss = y * dy.astype(y.dtype)
    return jnp.sum(loss)
  
  pjit_step_fn = pjit(value_and_grad(loss_fn, argnums=[0]))
  
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

    rules = (('batch', 'data'),)

    hlo_text = get_hlo_text(rules)

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
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[8192,4096]', # output
              'all-reduce',
              'f32[8192,4096]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on output dk for matmul")


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
