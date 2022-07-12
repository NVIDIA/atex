# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

"""Use fused instance norm ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader


_LAYOUT_OPTIMIZER_SUPPORTED_FORMATS = frozenset({
    "NCHW", "NHWC", "NCDHW","NDHWC"
})


norm_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_fused_nv_norm_ops.so'))
fused_instance_norm_grad = norm_ops.fused_instance_norm_grad
fused_layer_norm = norm_ops.fused_layer_norm
fused_layer_norm_grad = norm_ops.fused_layer_norm_grad


def fused_instance_norm(x, gamma, beta, data_format=None, name=None):
  with ops.name_scope(name, "FusedInstanceNorm", [x, gamma, beta]) as name:
    if data_format not in _LAYOUT_OPTIMIZER_SUPPORTED_FORMATS:
      if data_format is not None:
        if data_format.startswith("NC"):
          data_format = "NCHW"
        elif data_format.startswith("N") and data_format.endswith("C"):
          data_format = "NHWC"
        else:
          raise ValueError("`data_format` must be of the form `N...C` or "
                           f"`NC...`. Received: data_format={data_format}")
      else:
        data_format = "NHWC"

    if not context.executing_eagerly():
      x = ops.convert_to_tensor(x, name="x")
      gamma = ops.convert_to_tensor(gamma, name="gamma")
      beta = ops.convert_to_tensor(beta, name="beta")

  return norm_ops.fused_instance_norm(x, gamma, beta, data_format=data_format)

@ops.RegisterGradient("FusedInstanceNorm")
def _instance_norm_grad(op, *grad):
  """The gradients for `fused_instance_norm`.

  Args:
    op: The `fused_instance_norm` `Operation` that we are differentiating, which we
      can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `fused_instance_norm` op.

  Returns:
    Gradients with respect to the input of `fused_instance_norm`.
  """
  x = op.inputs[0]
  gamma = op.inputs[1]
  a = op.outputs[1]
  b = op.outputs[2]

  dx, dgamma, dbeta = norm_ops.fused_instance_norm_grad(
      grad[0], x, gamma, a, b, data_format=op.get_attr("data_format"))
  return [dx, dgamma, dbeta]

@ops.RegisterGradient("FusedLayerNorm")
def _layer_norm_grad(op, *grad):
  """The gradients for `fused_layer_norm`.

  Args:
    op: The `fused_layer_norm` `Operation` that we are differentiating, which we
      can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `fused_layer_norm` op.

  Returns:
    Gradients with respect to the input of `fused_layer_norm`.
  """
  x = op.inputs[0]
  gamma = op.inputs[1]
  a = op.outputs[1]
  b = op.outputs[2]

  dx, dgamma, dbeta = fused_layer_norm_grad(grad[0], x, gamma, a, b)
  return [dx, dgamma, dbeta]
