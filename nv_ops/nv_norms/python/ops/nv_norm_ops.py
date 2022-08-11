# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

"""Use fused layer and instance norm ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader


norm_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_fused_nv_norm_ops.so'))
fused_instance_norm_op = norm_ops.fused_instance_norm
fused_instance_norm_grad_op = norm_ops.fused_instance_norm_grad
fused_layer_norm_op = norm_ops.fused_layer_norm
fused_layer_norm_grad_op = norm_ops.fused_layer_norm_grad

@ops.RegisterGradient("FusedInstanceNorm")
def _instance_norm_grad(op, *grad):
  """The gradients for `fused_instance_norm`.

  Args:
    op: The `fused_instance_norm` `Operation` that we are differentiating, which
      we can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `fused_instance_norm` op.

  Returns:
    Gradients with respect to the input of `fused_instance_norm`.
  """
  x = op.inputs[0]
  gamma = op.inputs[1]
  a = op.outputs[1]
  b = op.outputs[2]

  dx, dgamma, dbeta = fused_instance_norm_grad_op(
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

  dx, dgamma, dbeta = fused_layer_norm_grad_op(
      grad[0], x, gamma, a, b, axis=op.get_attr("axis"))
  return [dx, dgamma, dbeta]

class LayerNormalization(tf.keras.layers.Layer):
  """LayerNormalization Layer.
    Args: Same with tf.keras.layers.LayerNormalization except that axis has to
      be packed and include last dimension.
    Output shape:
      y: Same shape as input.
  """

  def __init__(self, **kwargs):
    super(LayerNormalization, self).__init__()
    self.layer_norm = tf.keras.layers.LayerNormalization(**kwargs)

  def build(self, input_shape):
    self.layer_norm.build(input_shape=input_shape)
    self.built = True

  def call(self, inputs):
    axis = self.layer_norm.axis
    # Nv norm ops require the axis to be a list.
    if isinstance(axis, int):
      axis = [axis]
    if axis != sorted(set(axis)):
      raise ValueError('We only support sorted and unique axis to make sure '
                       'the weights have the same data layout with the keras '
                       'layers.')
    y, _, _ = fused_layer_norm_op(inputs,
                                  self.layer_norm.gamma,
                                  self.layer_norm.beta,
                                  axis=axis,
                                  epsilon=self.layer_norm.epsilon)
    return y

  def get_weights(self):
    return self.layer_norm.get_weights()

  def set_weights(self, weights):
    self.layer_norm.set_weights(weights)

  @property
  def variables(self):
    """Returns the list of all layer variables/weights.
    Alias of `self.weights`.
    Returns:
      A list of variables.
    """
    return self.layer_norm.weights

  def get_config(self):
    config = {
        'layer_norm': self.layer_norm,
    }
    base_config = super(LayerNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class InstanceNormalization(tf.keras.layers.Layer):
  """InstanceNormalization Layer.
    Args: Same with tfa.layer.InstanceNormalization except that axis only takes 
      value form -1 and 1.
    Output shape:
      y: Same shape as input.
  """
  def __init__(self, axis, **kwargs):
    super(InstanceNormalization, self).__init__()
    policy = tf.keras.mixed_precision.global_policy()
    is_mixed_policy = (
        policy is not None and policy.compute_dtype != policy.variable_dtype
    )
    # The FusedInstanceNorm requires the fp32 weights. So, we explicitly use the
    # "float32" policy to avoid the weight autocasting in the "mixed_float16"
    # scenario.
    if is_mixed_policy:
      tf.keras.mixed_precision.set_global_policy("float32")
    self.instance_norm = tfa.layers.InstanceNormalization(axis=axis,**kwargs)
    if is_mixed_policy:
      tf.keras.mixed_precision.set_global_policy(policy)

  def build(self, input_shape):
    self.instance_norm.build(input_shape=input_shape)
    self.built = True

  def call(self, inputs):
    axis = self.instance_norm.axis
    # Nv norm ops require the data format instead of axis.
    if axis == 1:
      data_format = "NC..."
    elif axis == -1:
      data_format = "N...C"
    else:
      raise ValueError('We only support integer axis of 1 or -1 corresponds to' 
                       'channel first or channel last layout.')
    y, _, _ = fused_instance_norm_op(inputs,
                                     self.instance_norm.weights[0],
                                     self.instance_norm.weights[1],
                                     data_format=data_format,
                                     epsilon=self.instance_norm.epsilon)
    return y

  def get_config(self):
    config = {'instance_norm': self.instance_norm}
    base_config = super(InstanceNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_weights(self):
    return self.instance_norm.get_weights()

  def set_weights(self, weights):
    self.instance_norm.set_weights(weights)

  @property
  def variables(self):
    """Returns the list of all layer variables/weights.
    Alias of `self.weights`.
    Returns:
      A list of variables.
    """
    return self.instance_norm.weights

