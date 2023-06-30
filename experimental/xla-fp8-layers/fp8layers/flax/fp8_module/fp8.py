from functools import partial

from jax import custom_vjp
from jax import lax
from jax import numpy as jnp

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

@partial(custom_vjp, nondiff_argnums=(0,))
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


@partial(custom_vjp, nondiff_argnums=(0,))
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
