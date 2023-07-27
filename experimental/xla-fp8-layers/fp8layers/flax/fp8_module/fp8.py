from functools import partial
import numpy as np

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

def fp8_dot(x, k, compute_dtype, x_scale, x_amax_history, k_scale,
            k_amax_history, dy_scale, dy_amax_history):
  """Perform einsum formula 'MK,KN->MN'. """
  assert len(x.shape) == 2, f'x rank has to be 2, but got {len(x.shape)}'
  assert len(k.shape) == 2, f'k rank has to be 2, but got {len(k.shape)}'

  x_qdq = in_qdq(compute_dtype, x, x_scale, x_amax_history)

  k_qdq = in_qdq(compute_dtype, k, k_scale, k_amax_history)

  y_qdq = lax.dot(x_qdq, k_qdq)

  y = out_qdq(compute_dtype, y_qdq, dy_scale, dy_amax_history)

  return y

def fp8_projection(x, w, use_bias, b, x_scale, x_amax_history, w_scale,
                   w_amax_history, dy_scale, dy_amax_history):
  """Perform einsum formula '...K,KN->...N'. """
  assert len(x.shape) > 1, f'x rank has to be > 1, but got {len(x.shape)}'
  assert len(w.shape) == 2, f'w rank has to be 2, but got {len(w.shape)}'
  assert len(b.shape) == 1, f'b rank has to be 1, but got {len(b.shape)}'

  batch_dims_rank = len(x.shape) - 1
  dtype = x.dtype

  batch_shape = x.shape[0:-1]
  n_dim = w.shape[-1]

  # ...K->(...)K
  x_mat = jnp.reshape(x, (-1, x.shape[-1]))

  # (...)K,KN->(...)N
  y_mat = fp8_dot(x_mat, w, dtype, x_scale, x_amax_history, w_scale,
                  w_amax_history, dy_scale, dy_amax_history)

  if use_bias:
    if b.dtype == jnp.float32:
      b = b.astype(jnp.bfloat16)
      b = b.astype(jnp.float32)
    y_mat = y_mat + b

  y_shape = (*batch_shape, n_dim)
  # (...)N->...N
  y = jnp.reshape(y_mat, y_shape)
  return y

def fp8_qkv_combined_projection(x, w, use_bias, b, x_scale, x_amax_history,
                                w_scale, w_amax_history, dy_scale,
                                dy_amax_history):
  """Perform einsum formula '...D,KDNH->K...NH'. """
  assert len(x.shape) > 1, f'x rank has to be > 1, but got {len(x.shape)}'
  assert len(w.shape) == 4, f'w rank has to be 4, but got {len(w.shape)}'

  batch_dims_rank = len(x.shape) - 1
  dtype = x.dtype

  batch_shape = x.shape[0:-1]
  knh_shape = [w.shape[0]] + list(w.shape[2:])

  # ...D->(...)D
  x_mat = jnp.reshape(x, (-1, x.shape[-1]))
  # KDNH->DKNH
  w_mat = jnp.transpose(w, (1, 0, 2, 3))
  # DKNH->D(KNH)
  w_mat = jnp.reshape(w_mat, (-1, np.prod(knh_shape)))
  # KNH->(KNH)
  b_vec = jnp.reshape(b, (-1,))

  # (...)D,D(KNH)->(...)(KNH)
  y_mat = fp8_dot(x_mat, w_mat, dtype, x_scale, x_amax_history, w_scale,
                  w_amax_history, dy_scale, dy_amax_history)
  if use_bias:
    if b_vec.dtype == jnp.float32:
      b_vec = b_vec.astype(jnp.bfloat16)
      b_vec = b_vec.astype(jnp.float32)
    y_mat = y_mat + b_vec
  
  y_shape = (*batch_shape, *knh_shape)
  # (...)(KNH)->...KNH
  y = jnp.reshape(y_mat, y_shape)
  permute = list(range(len(y_shape)))
  permute = [permute[batch_dims_rank]] + permute[0:batch_dims_rank] + \
            permute[batch_dims_rank + 1:]
  # ...KNH->K...NH
  y = jnp.transpose(y, permute)
  return y

def fp8_qkv_projection(x, w, use_bias, b, x_scale, x_amax_history, w_scale,
                       w_amax_history, dy_scale, dy_amax_history):
  """Perform einsum formula '...D,DNH->...NH'. """
  assert len(x.shape) > 1, f'x rank has to be > 1, but got {len(x.shape)}'
  assert len(w.shape) == 3, f'w rank has to be 3, but got {len(w.shape)}'

  batch_dims_rank = len(x.shape) - 1
  dtype = x.dtype

  batch_shape = x.shape[0:-1]
  nh_shape = list(w.shape[1:])

  # ...D->(...)D
  x_mat = jnp.reshape(x, (-1, x.shape[-1]))
  # DNH->D(NH)
  w_mat = jnp.reshape(w, (-1, np.prod(nh_shape)))
  # NH->(NH)
  b_vec = jnp.reshape(b, (-1,))

  # (...)D,D(NH)->(...)(NH)
  y_mat = fp8_dot(x_mat, w_mat, dtype, x_scale, x_amax_history, w_scale,
                  w_amax_history, dy_scale, dy_amax_history)
  if use_bias:
    if b_vec.dtype == jnp.float32:
      b_vec = b_vec.astype(jnp.bfloat16)
      b_vec = b_vec.astype(jnp.float32)
    y_mat = y_mat + b_vec
  
  y_shape = (*batch_shape, *nh_shape)
  # (...)(NH)->...NH
  y = jnp.reshape(y_mat, y_shape)
  return y

def fp8_attention_output_projection(x, w, use_bias, b, x_scale, x_amax_history,
                                    w_scale, w_amax_history, dy_scale,
                                    dy_amax_history, use_nhd_shape):
  """Perform einsum formula '...NH,NHD->...D' or '...NH,DNH->...D'. """
  assert len(x.shape) > 2, f'x rank has to be > 2, but got {len(x.shape)}'
  assert len(w.shape) == 3, f'w rank has to be 3, but got {len(w.shape)}'
  assert len(b.shape) == 1, f'b rank has to be 1, but got {len(b.shape)}'

  batch_dims_rank = len(x.shape) - 2
  dtype = x.dtype

  batch_shape = x.shape[0:-2]

  # ...NH->(...)(NH)
  x_mat = jnp.reshape(x, (-1, np.prod(x.shape[-2:])))
  if use_nhd_shape:
    d_dim = w.shape[-1]
    # NHD->(NH)D
    w_mat = jnp.reshape(w, (-1, w.shape[-1]))
  else:
    d_dim = w.shape[0]
    # DNH->NHD
    w_mat = jnp.transpose(w, (1, 2, 0))
    # NHD->(NH)D
    w_mat = jnp.reshape(w_mat, (-1, w_mat.shape[-1]))

  # (...)(NH),(NH)D->(...)D
  y_mat = fp8_dot(x_mat, w_mat, dtype, x_scale, x_amax_history, w_scale,
                  w_amax_history, dy_scale, dy_amax_history)
  if use_bias:
    if b.dtype == jnp.float32:
      b = b.astype(jnp.bfloat16)
      b = b.astype(jnp.float32)
    y_mat = y_mat + b
  
  y_shape = (*batch_shape, d_dim)
  # (...)D->...D
  y = jnp.reshape(y_mat, y_shape)
  return y



