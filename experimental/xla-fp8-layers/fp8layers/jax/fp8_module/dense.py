from functools import partial
from typing import Callable, Iterable, Optional, Dict, Union, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax import core
from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.linen import partitioning as nn_partitioning
from flax.linen import spmd
from jax import lax

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

class FP8Helper:
  FP8_COLLECTION_NAME: str = "fp8_params"

  @staticmethod
  def update_fp8_params(state: Collection, grads: Collection) -> Collection:
    """
    Update the FP8 params
    """
    if FP8Helper.FP8_COLLECTION_NAME in state:
      if not isinstance(state, FrozenDict):
        state = FrozenDict(state)
      others, fp8_params = state.pop(FP8Helper.FP8_COLLECTION_NAME)
      new_fp8_params = grads[FP8Helper.FP8_COLLECTION_NAME]
      return FrozenDict({**others,
                         FP8Helper.FP8_COLLECTION_NAME : new_fp8_params})
    return state

class TrainState(struct.PyTreeNode):
  """
  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    params: The parameters contains two parts: "fp8_params" collections and
      others. During the apply_gradients, parameters in the "fp8_params" will be
      directly replaced by the corresponding grads; other parameters will be
      updated by `tx`.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
  """
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, **kwargs):
    others_grads, fp8_grads = grads.pop(FP8Helper.FP8_COLLECTION_NAME)
    others, _ = self.params.pop(FP8Helper.FP8_COLLECTION_NAME)

    updates, new_opt_state = self.tx.update(
        others_grads, self.opt_state, others)
    new_params = optax.apply_updates(others, updates)

    return self.replace(
        step=self.step + 1,
        params=FrozenDict({**new_params,
                           FP8Helper.FP8_COLLECTION_NAME : fp8_grads}),
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    others, _ = params.pop(FP8Helper.FP8_COLLECTION_NAME)
    opt_state = tx.init(others)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )

def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])

def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)

def get_fp8_max(fp8_dtype):
  assert fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
  return jnp.finfo(fp8_dtype).max.astype(jnp.float32)

def quantize(x, q_dtype, scale, compute_dtype):
  # We need to explicitly cast the max value to compute_dtype, otherwise the jax
  # dtype promotion will cast the scaled_x to fp32 in the following ops, which
  # would violate the fp8-matmul pattern matching.
  dtype_max = get_fp8_max(q_dtype).astype(compute_dtype)

  scaled_x = x / scale.astype(compute_dtype)
  clipped_x = jnp.clip(scaled_x, -dtype_max, dtype_max)

  return clipped_x.astype(q_dtype)

def dequantize(x, dq_dtype, scale):
  return x.astype(dq_dtype) * scale.astype(dq_dtype)

def quantize_dequantize(x, q_dtype, scale, compute_dtype):
  qx = quantize(x, q_dtype, scale, compute_dtype)
  return dequantize(qx, x.dtype, scale)

def compute_scale(amax, scale, fp8_max, margin=0):
  """Default function to convert amax to scaling factor."""
  exp = jnp.floor(jnp.log2(fp8_max / amax)) - margin
  sf = jnp.round(lax.pow(2., jnp.abs(exp)))
  sf = jnp.where(amax > 0.0, sf, scale)
  sf = jnp.where(lax.is_finite(amax), sf, scale)
  sf = jnp.where(exp < 0, 1.0 / sf, sf)
  # The scaling factor we need equals to the notion of "scale_inv" in
  # TransformerEngine. So, we convert the sf to its reciprocal.
  return 1.0 / sf

def compute_scale_and_amax_history(x, q_dtype, scale, amax_history):
  dtype_max = get_fp8_max(q_dtype)

  amax_update = jnp.max(jnp.abs(x)).astype(scale.dtype)
  new_amax_history = \
      jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)

  amax_from_history = jnp.max(new_amax_history, axis=0)
  new_scale = compute_scale(amax_from_history, scale, dtype_max)
  return new_scale, new_amax_history

def qdq_and_return(x, q_dtype, scale, amax_history, compute_dtype):
  qx = quantize_dequantize(x, q_dtype, scale, compute_dtype)
  new_scale, new_amax_history = compute_scale_and_amax_history(
      x, q_dtype, scale, amax_history)
  return qx, new_scale, new_amax_history

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def in_qdq(compute_dtype, inp, scale, amax_history):
  qin, _, _ = qdq_and_return(
      inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
  return qin

def in_qdq_fwd(compute_dtype, inp, scale, amax_history):
  qin, new_scale, new_amax_history = qdq_and_return(
      inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
  return qin, (new_scale, new_amax_history)

def in_qdq_bwd(compute_dtype, res, g):
  new_scale, new_amax_history = res
  q_g = g
  return q_g, new_scale, new_amax_history

in_qdq.defvjp(in_qdq_fwd, in_qdq_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def out_qdq(compute_dtype, out, scale, amax_history):
  return out

def out_qdq_fwd(compute_dtype, out, scale, amax_history):
  return out, (scale, amax_history)

def out_qdq_bwd(compute_dtype, res, g):
  scale, amax_history = res
  q_g, new_scale, new_amax_history = qdq_and_return(
      g, jnp.float8_e5m2, scale, amax_history, compute_dtype)
  return q_g, new_scale, new_amax_history
  
out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)

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
        jax.random.PRNGKey(0),
        (1,),
        jnp.float32,
    )
    amax_history_args = (
        nn.initializers.zeros_init(),
        jax.random.PRNGKey(0),
        (self.amax_history_length,),
        jnp.float32,
    )

    input_amax_history = self.variable(
        FP8Helper.FP8_COLLECTION_NAME, 'input_amax_history', *amax_history_args)
    kernel_amax_history = self.variable(
        FP8Helper.FP8_COLLECTION_NAME, 'kernel_amax_history',
        *amax_history_args)
    output_grad_amax_history = self.variable(
        FP8Helper.FP8_COLLECTION_NAME, 'output_grad_amax_history',
        *amax_history_args)

    input_scale = self.variable(
        FP8Helper.FP8_COLLECTION_NAME, 'input_scale', *scale_args)
    kernel_scale = self.variable(
        FP8Helper.FP8_COLLECTION_NAME, 'kernel_scale', *scale_args)
    output_grad_scale = self.variable(
        FP8Helper.FP8_COLLECTION_NAME, 'output_grad_scale', *scale_args)

    inputs, kernel, bias = nn.dtypes.promote_dtype(inputs, kernel, bias,
                                                   dtype=self.dtype)

    # Reshape the inputs to 2D matrix.
    inp_mat = jnp.reshape(inputs, (-1, original_shape[-1]))

    inp_mat = in_qdq(self.dtype, inp_mat, input_scale.value,
                     input_amax_history.value)
    kernel = in_qdq(self.dtype, kernel, kernel_scale.value,
                    kernel_amax_history.value)

    # Actual dense layer math.
    out = lax.dot(inp_mat, kernel)

    out = out_qdq(self.dtype, out, output_grad_scale.value,
                  output_grad_amax_history.value)

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
    out = jnp.reshape(out, (*original_shape[0:-len(axis)], out.shape[-1]))
  
    return out



