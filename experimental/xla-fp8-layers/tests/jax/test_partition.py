"""Tests for the fp8 layers with partitioning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from functools import partial
import re

import optax
import os

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
from jax import lax
from jax import random

import flax
from flax import linen as nn
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict

from fp8layers.jax import DenseGeneral, FP8Helper, TrainState

# Sharding related
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils
from flax import struct, traverse_util, linen as nn
from flax.linen import spmd # Flax Linen SPMD.


@jtu.with_config(jax_numpy_rank_promotion='allow',
                 jax_numpy_dtype_promotion='standard')
class PartitionTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

  def testAllReduceDpTpCol(self):
    if jtu.device_under_test() != "gpu" or jax.device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    device_mesh = mesh_utils.create_device_mesh((4, 2))
    mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
    
    model = DenseGeneral(8192, use_bias=False, kernel_axes=('embed', 'mlp'))
    
    x = random.normal(random.PRNGKey(0), (8192, 8192))
    dy = random.normal(random.PRNGKey(0), (8192, 8192))
    k = random.PRNGKey(0)
    
    # A functional way of model initialization.
    def init_fn(k, x):
      variables = model.init(k, x) # Initialize the model.
      return variables
    
    abstract_variables = jax.eval_shape(init_fn, k, x)
    logical_output_spec = nn.get_partition_spec(abstract_variables)
    
    rules = (('batch', 'data'),
             ('mlp', 'model'))
    
    logical_state_spec = spmd.logical_to_mesh(logical_output_spec, rules)
    
    pjit_init_fn = pjit(
        init_fn,
        in_axis_resources=(PartitionSpec(None),
                           PartitionSpec('data', None)),  # PRNG key and x
        out_axis_resources=logical_state_spec,  # params
    )

    with mesh:
      initialized_state = pjit_init_fn(k, x)
    
    def loss_fn(state, x, dy):
      y = model.apply(state, x)
      loss = y * dy.astype(y.dtype)
      return jnp.sum(loss)
    
    pjit_step_fn = pjit(
        jax.value_and_grad(loss_fn, argnums=[0]),
        in_axis_resources=(
            logical_state_spec,
            PartitionSpec('data', None),
            PartitionSpec('data', 'model'),
        ),
    )
    
    with mesh:
      lowered = pjit_step_fn.lower(initialized_state, x, dy)
    hlo = lowered.compile()
    self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on kernel amax")
    self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on input amax")
    # The loss function applies sum on the output, which will also generate an
    # all-reduce.
    # TODO(kaixih): find a way to differentiate these two cases.
    self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,1,2,3,4,5,6,7}}',
          )])), msg="all-reduce on output_grad amax")

  def testAllReduceDpTpRow(self):
    if jtu.device_under_test() != "gpu" or jax.device_count() < 8:
      self.skipTest(f"Test enabled only for 8 GPUs")

    device_mesh = mesh_utils.create_device_mesh((4, 2))
    mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
    
    model = DenseGeneral(8192, use_bias=False, kernel_axes=('embed', 'mlp'))
    
    x = random.normal(random.PRNGKey(0), (8192, 8192))
    dy = random.normal(random.PRNGKey(0), (8192, 8192))
    k = random.PRNGKey(0)
    
    # A functional way of model initialization.
    def init_fn(k, x):
      variables = model.init(k, x) # Initialize the model.
      return variables
    
    abstract_variables = jax.eval_shape(init_fn, k, x)
    logical_output_spec = nn.get_partition_spec(abstract_variables)
    
    rules = (('batch', 'data'),
             ('embed', 'model'))
    
    logical_state_spec = spmd.logical_to_mesh(logical_output_spec, rules)
    
    pjit_init_fn = pjit(
        init_fn,
        in_axis_resources=(PartitionSpec(None),
                           PartitionSpec('data', 'model')),  # PRNG key and x
        out_axis_resources=logical_state_spec,  # params
    )

    with mesh:
      initialized_state = pjit_init_fn(k, x)
    
    def loss_fn(state, x, dy):
      y = model.apply(state, x)
      loss = y * dy.astype(y.dtype)
      return jnp.sum(loss)
    
    pjit_step_fn = pjit(
        jax.value_and_grad(loss_fn, argnums=[0]),
        in_axis_resources=(
            logical_state_spec,
            PartitionSpec('data', 'model'),
            PartitionSpec('data', None),
        ),
    )
    
    with mesh:
      lowered = pjit_step_fn.lower(initialized_state, x, dy)
    hlo = lowered.compile()
    self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,1},{2,3},{4,5},{6,7}}',
          )])), msg="all-reduce on kernel amax")
    self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,2,4,6},{1,3,5,7}}',
          )])), msg="all-reduce on output_grad amax")
    # The loss function applies sum on the output, which will also generate an
    # all-reduce.
    # TODO(kaixih): find a way to differentiate these two cases.
    self.assertRegex(
          hlo.as_text(), re.compile('.*'.join([re.escape(x) for x in (
              'all-reduce',
              'f32[]', # output
              'all-reduce',
              'f32[]', # input
              'replica_groups={{0,1,2,3,4,5,6,7}}',
          )])), msg="all-reduce on input amax")


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
