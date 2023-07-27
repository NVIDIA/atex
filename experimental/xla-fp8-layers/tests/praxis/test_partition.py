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
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from flax.linen import spmd # Flax Linen SPMD.

from fp8layers.praxis import Linear

from praxis import base_layer
from praxis import pax_fiddle

instantiate = base_layer.instantiate
var_partition_specs = base_layer.var_partition_specs 
WeightSharding = base_layer.BaseLayer.WeightSharding

def get_hlo_text(in_shardings):
  mesh_names = ('data', 'model')
  mesh_shape = (4, 2)
  device_mesh = mesh_utils.create_device_mesh(mesh_shape)
  mesh = Mesh(devices=device_mesh, axis_names=mesh_names)

  linear_kwargs = {'input_dims': 8192, 'output_dims': 8192}
  model: Linear = instantiate(
      pax_fiddle.Config(Linear, name='linear_fp8', **linear_kwargs)
  )
  
  x = random.normal(random.PRNGKey(0), (8192, 8192))
  dy = random.normal(random.PRNGKey(0), (8192, 8192))
  k = random.PRNGKey(0)

  var_hparams = model.abstract_init_with_metadata(x)
  var_pspecs = var_partition_specs(var_hparams, mesh_shape, mesh_names)

  # in_shardings[0] only contains the PSpec of the weight. We also need to
  # define the PSpecs for all the other variables.
  var_pspecs['params']['w'] = in_shardings[0]
  in_shardings[0] = var_pspecs
  
  initialized_state = model.init(k, x)

  def loss_fn(state, x, dy):
    y = model.apply(state, x)
    loss = y * dy.astype(y.dtype)
    return jnp.sum(loss)
  
  pjit_step_fn = pjit(value_and_grad(loss_fn, argnums=[0]),
                      in_shardings=in_shardings)
  
  with mesh:
    lowered = pjit_step_fn.lower(initialized_state, x, dy)
  hlo = lowered.compile()
  return hlo.as_text()


@jtu.with_config(jax_numpy_rank_promotion='allow',
                 jax_numpy_dtype_promotion='standard')
class PartitionTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

  def testDp(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    in_shardings = [None, P('data', None), P('data', None)]
    hlo_text = get_hlo_text(in_shardings)

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
              'cublas',
              'f32[2048,8192]', # output
              'custom-call',
              'f8e4m3fn[2048,8192]{1,0}',
              'f8e4m3fn[8192,8192]{1,0}',
          )])), msg="fprop gemm")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f32[8192,8192]', # output
              'custom-call',
              'f8e4m3fn[8192,2048]{1,0}',
              'f8e5m2[8192,2048]{1,0}',
          )])), msg="bprop gemm")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[8192,8192]', # output
              'all-reduce',
              'f32[8192,8192]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on output results of dy*x=dk matmul")


  def testTpRow(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    in_shardings = [P('model', None), None, None]
    hlo_text = get_hlo_text(in_shardings)

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
              'cublas',
              'f32[8192,8192]', # output
              'custom-call',
              'f8e4m3fn[8192,4096]{1,0}',
              'f8e4m3fn[8192,4096]{1,0}',
          )])), msg="fprop gemm")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[8192,8192]', # output
              'all-reduce',
              'f32[8192,8192]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on intermediate results of x*k=y matmul")
  
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f32[4096,8192]', # output
              'custom-call',
              'f8e4m3fn[4096,8192]{1,0}',
              'f8e5m2[8192,8192]{1,0}',
          )])), msg="bprop gemm")
    

  def testTpCol(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    in_shardings = [P(None, 'model'), None, P(None, 'model')]
    hlo_text = get_hlo_text(in_shardings)

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

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f32[8192,4096]', # output
              'custom-call',
              'f8e4m3fn[8192,8192]{1,0}',
              'f8e4m3fn[4096,8192]{1,0}',
          )])), msg="fprop gemm")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f32[8192,4096]', # output
              'custom-call',
              'f8e4m3fn[8192,8192]{1,0}',
              'f8e5m2[4096,8192]{1,0}',
          )])), msg="bprop gemm")


  def testDpTpRow(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    in_shardings = [P('model', None), P('data', None), P('data', None)]
    hlo_text = get_hlo_text(in_shardings)
    
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
              'cublas',
              'f32[2048,8192]', # output
              'custom-call',
              'f8e4m3fn[2048,4096]{1,0}',
              'f8e4m3fn[8192,4096]{1,0}',
          )])), msg="fprop gemm")
    
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[2048,8192]', # output
              'all-reduce',
              'f32[2048,8192]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on output results of x*k=y matmul")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f32[4096,8192]', # output
              'custom-call',
              'f8e4m3fn[4096,2048]{1,0}',
              'f8e5m2[8192,2048]{1,0}',
          )])), msg="bprop gemm")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[4096,8192]', # output
              'all-reduce',
              'f32[4096,8192]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on intermediate results of dy*x=dk matmul")


  def testDpTpCol(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    in_shardings = [P(None, 'model'), P('data', None), P('data', 'model')]
    hlo_text = get_hlo_text(in_shardings)

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
              'cublas',
              'f32[2048,4096]', # output
              'custom-call',
              'f8e4m3fn[2048,8192]{1,0}',
              'f8e4m3fn[4096,8192]{1,0}',
          )])), msg="fprop gemm")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'cublas',
              'f32[8192,4096]', # output
              'custom-call',
              'f8e4m3fn[8192,2048]{1,0}',
              'f8e5m2[4096,2048]{1,0}',
          )])), msg="bprop gemm")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[8192,4096]', # output
              'all-reduce',
              'f32[8192,4096]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on output results of dy*x=dk matmul")


  def testFullShardingDpTpRow(self):
    # In the typical Dp test, we only shard the input tensor by its first axis.
    # This test, however, shard the input tensor along both axes. Eventually,
    # this test same with the testDpTpRow, but we avoid the slicing on the input
    # tensor.

    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    in_shardings = [P('model', None), P('data', 'model'), P('data', None)]
    hlo_text = get_hlo_text(in_shardings)
    
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
              'cublas',
              'f32[2048,8192]', # output
              'custom-call',
              'f8e4m3fn[2048,4096]{1,0}',
              'f8e4m3fn[8192,4096]{1,0}',
          )])), msg="fprop gemm")
    
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
              'cublas',
              'f32[4096,8192]', # output
              'custom-call',
              'f8e4m3fn[4096,2048]{1,0}',
              'f8e5m2[8192,2048]{1,0}',
          )])), msg="bprop gemm")

    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[4096,8192]', # output
              'all-reduce',
              'f32[4096,8192]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on intermediate results of dy*x=dk matmul")


  def testFullSharding(self):
    if jtu.device_under_test() != "gpu" or device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    in_shardings = [P('data', 'model'), P('data', 'model'), P('data', 'model')]
    hlo_text = get_hlo_text(in_shardings)

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

    # TODO(shuw): check if the fp8 gemm is used when the cases of all-gather
    # over inputs get supported.
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'custom-call',
              'f32[2048,4096]', # output
              'custom-call',
              'f32[2048,8192]{0,1}',
              'f32[8192,4096]{1,0}',
          )])), msg="fprop gemm")

    # TODO(shuw): check if the fp8 gemm is used when the cases of all-gather
    # over inputs get supported.
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'custom-call',
              'f32[4096,8192]', # output
              'custom-call',
              'f32[2048,4096]{0,1}',
              'f32[2048,8192]{0,1}',
          )])), msg="bprop gemm")
    
    self.assertRegex(
          hlo_text, re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[4096,8192]', # output
              'all-reduce',
              'f32[4096,8192]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on output dk for matmul")


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
