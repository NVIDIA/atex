from typing import Callable, Iterable, Optional, Dict, Union, Any, Tuple

import numpy as np
import optax

from flax import core
from flax import linen as nn
from flax import traverse_util
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.linen import partitioning as nn_partitioning
from jax import lax
from jax import numpy as jnp

from .fp8 import in_qdq, out_qdq

# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

# Type annotations
Array = jnp.ndarray
Dtype = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
ActivationFn = Callable[..., Array]
DotGeneralT = Callable[..., Array]
Collection = Union[Dict, FrozenDict]

def _validate_params_axes(params_axes, params):
  axis_names = nn_partitioning.get_axis_names(params_axes)
  missing_params_axes = (
      set(traverse_util.flatten_dict(params, sep='/')) -
      set(traverse_util.flatten_dict(axis_names, sep='/')))
  if missing_params_axes:
    raise ValueError(
        f'Missing axis names for parameters: {missing_params_axes}')

def _split_fp8_and_others(params):
  flt_fp8 = {}
  flt_other = {}
  flt_params = traverse_util.flatten_dict(params, sep='/')
  for k, v in flt_params.items():
    if k.endswith('_fp8_meta'):
      flt_fp8[k] = v
    else:
      flt_other[k] = v
  fp8_params = traverse_util.unflatten_dict(flt_fp8, sep='/')
  other_params = traverse_util.unflatten_dict(flt_other, sep='/')
  return core.freeze(fp8_params), core.freeze(other_params)

def _merge_fp8_and_others(fp8_params, others):
  flt_fp8 = traverse_util.flatten_dict(fp8_params, sep='/')
  flt_other = traverse_util.flatten_dict(others, sep='/')
  flt_params = {**flt_fp8, **flt_other}
  return traverse_util.unflatten_dict(flt_params, sep='/')

class TrainState(struct.PyTreeNode):
  """
  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    params: The params that will be updated by the `tx`.
    fp8_params: The fp8_meta params that will be replaced by their grads.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
    params_axes: Contains axis metadata (e.g., names) matching `params` tree.
  """
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  params_axes: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  fp8_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  
  def variables(self) -> core.FrozenDict[str, Any]:
    variables = {}
    variables['params'] = _merge_fp8_and_others(self.fp8_params, self.params)
    return core.freeze(variables)

  def apply_gradients(self, *, grads, **kwargs):
    fp8_grads, other_grads = _split_fp8_and_others(grads['params'])

    updates, new_opt_state = self.tx.update(
        other_grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)

    return self.replace(
        step=self.step + 1,
        params=new_params,
        fp8_params=fp8_grads,
        opt_state=new_opt_state,
    )

  @classmethod
  def create(cls, apply_fn, model_variables, tx):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    other_variables, params = core.pop(model_variables, 'params')
    fp8_params, other_params = _split_fp8_and_others(params)

    if 'params_axes' in other_variables:
      other_variables, params_axes = core.pop(
          other_variables, 'params_axes'
      )
      _validate_params_axes(params_axes, other_params)
    else:
      params_axes = None

    if len(other_variables) > 0:
      raise ValueError(f'Contains unknown variables: {other_variables.keys}')

    opt_state = tx.init(other_params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=other_params,
        fp8_params=fp8_params,
        tx=tx,
        opt_state=opt_state,
        params_axes=params_axes,
    )

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
        (1,),
        jnp.float32,
    )
    amax_history_args = (
        nn.initializers.zeros_init(),
        (self.amax_history_length,),
        jnp.float32,
    )

    input_amax_history = self.param(
        'input_amax_history_fp8_meta', *amax_history_args)
    kernel_amax_history = self.param(
        'kernel_amax_history_fp8_meta', *amax_history_args)
    output_grad_amax_history = self.param(
        'output_grad_amax_history_fp8_meta', *amax_history_args)

    input_scale = self.param('input_scale_fp8_meta', *scale_args)
    kernel_scale = self.param('kernel_scale_fp8_meta', *scale_args)
    output_grad_scale = self.param('output_grad_scale_fp8_meta', *scale_args)

    inputs, kernel, bias = nn.dtypes.promote_dtype(inputs, kernel, bias,
                                                   dtype=self.dtype)

    # Reshape the inputs to 2D matrix.
    inp_mat = jnp.reshape(inputs,
                          (-1, np.prod([inputs.shape[ax] for ax in axis])))
    inp_mat = in_qdq(self.dtype, inp_mat, input_scale,
                     input_amax_history)
    kernel = in_qdq(self.dtype, kernel, kernel_scale,
                    kernel_amax_history)

    # Actual dense layer math.
    out = lax.dot(inp_mat, kernel)

    out = out_qdq(self.dtype, out, output_grad_scale,
                  output_grad_amax_history)

    if self.use_bias:
      # The bias has already been promoted. So, if it is fp32, we need to cast
      # it to bf16 to trigger fp8 matmul fusion.
      if bias.dtype == jnp.float32:
        bias = bias.astype(jnp.bfloat16)
        bias = bias.astype(jnp.float32)
      out = out + bias

    if self.activation:
      out = self.activation(out)

    # Reshape back the outputs.
    out = jnp.reshape(out, (*original_shape[0:-len(axis)], *tuple(features)))
  
    return out

