from typing import Callable, Iterable, Optional, Dict, Union, Tuple

import numpy as np

from flax import core
from flax.core import scope as flax_scope
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax
from jax import numpy as jnp
from jax import random

from .fp8 import fp8_dot

param_with_axes = nn_partitioning.param_with_axes
variable_with_axes = nn_partitioning.variable_with_axes

# Type annotations
Array = jnp.ndarray
Dtype = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
ActivationFn = Callable[..., Array]
DotGeneralT = Callable[..., Array]

class FP8Helper:
  FP8_COLLECTION_NAME: str = "fp8_params"

def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])

def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)

class DenseGeneral(nn.Module):
  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  amax_history_length: int = 16
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = \
      nn.initializers.lecun_normal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  activation: Optional[ActivationFn] = None
  dot_general: DotGeneralT = lax.dot_general
  kernel_axes: Tuple[str, ...] = ()
  bias_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    original_shape = inputs.shape
    assert len(original_shape) >= 2

    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                          np.prod(features))

    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_param_shape,
        self.param_dtype,
        axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)

    if self.use_bias:
      bias = param_with_axes(
          'bias',
          self.bias_init,
          (np.prod(features),),
          self.param_dtype,
          axes=self.bias_axes)
      bias = jnp.asarray(bias, self.dtype)
    else:
      bias = None

    scale_args = (
        nn.initializers.ones_init(),
        random.PRNGKey(0),
        (1,),
        jnp.float32,
    )
    amax_history_args = (
        nn.initializers.zeros_init(),
        random.PRNGKey(0),
        (self.amax_history_length,),
        jnp.float32,
    )

    input_amax_history = variable_with_axes(
        FP8Helper.FP8_COLLECTION_NAME,
        'input_amax_history',
        *amax_history_args,
        axes=('fp8_meta',))
    kernel_amax_history = variable_with_axes(
        FP8Helper.FP8_COLLECTION_NAME,
        'kernel_amax_history',
        *amax_history_args,
        axes=('fp8_meta',))
    output_grad_amax_history = variable_with_axes(
        FP8Helper.FP8_COLLECTION_NAME,
        'output_grad_amax_history',
        *amax_history_args,
        axes=('fp8_meta',))

    input_scale = variable_with_axes(
        FP8Helper.FP8_COLLECTION_NAME,
        'input_scale',
        *scale_args,
        axes=('fp8_meta',))
    kernel_scale = variable_with_axes(
        FP8Helper.FP8_COLLECTION_NAME,
        'kernel_scale',
        *scale_args,
        axes=('fp8_meta',))
    output_grad_scale = variable_with_axes(
        FP8Helper.FP8_COLLECTION_NAME,
        'output_grad_scale',
        *scale_args,
        axes=('fp8_meta',))


    inputs, kernel, bias = nn.dtypes.promote_dtype(inputs, kernel, bias,
                                                   dtype=self.dtype)

    # Reshape the inputs to 2D matrix to call the fp8_dot. The result will be
    # casted back at the end.
    inp = jnp.reshape(inputs, (-1, np.prod([inputs.shape[ax] for ax in axis])))

    out = fp8_dot(inp, kernel, self.dtype,
                  input_scale.value, input_amax_history.value,
                  kernel_scale.value, kernel_amax_history.value,
                  output_grad_scale.value, output_grad_amax_history.value)

    if self.use_bias:
      # To this point, the bias has already been promoted to self.dtype. If it
      # is still fp32, we manually cast it to bf16 to trigger fullfil the fp8
      # matmul fusion requirement.
      if bias.dtype == jnp.float32:
        bias = bias.astype(jnp.bfloat16)
        bias = bias.astype(jnp.float32)
      out = out + bias

    if self.activation:
      out = self.activation(out)

    # Reshape back the outputs.
    out = jnp.reshape(out, (*original_shape[0:-len(axis)], *tuple(features)))
  
    return out
