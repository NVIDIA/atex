import argparse
import time
from functools import partial
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import optax

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from fp8layers.flax import DenseGeneral, TrainState
from jax.experimental.pjit import pjit


param_with_axes = nn_partitioning.param_with_axes

dropout_rate = 0.0

parser = argparse.ArgumentParser(description='Benchmark a basic encoder layer')
parser.add_argument('--fp8', action='store_true', help='Enable fp8')
parser.add_argument('--mixed', action='store_true',
                    help='Enable mixed precision and bf16 compute type')
args = parser.parse_args()

use_fp8 = args.fp8
use_mixed = args.mixed
TrainState = TrainState if use_fp8 else train_state.TrainState

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Initializer = Callable[[PRNGKey, Shape, DType], Array]

# Note, in the jax examples, we use the bf16 in the mixed precision. This is
# different from the fp16 in the TF examples.
dtype = jnp.bfloat16 if use_mixed else jnp.float32

DenseLayer = DenseGeneral if use_fp8 else nn.DenseGeneral

class DotProductAttention(nn.Module):
  """Attention operation in Transformer layer"""
  num_attention_heads: int
  kv_channels: int
  attention_dropout: float

  def masked_softmax(
      self,
      inp: jnp.ndarray,
      mask: Optional[jnp.ndarray],
  ) -> jnp.ndarray:
    if mask is not None:
      inp = jnp.where(mask, -10000.0, inp)
    return jax.nn.softmax(inp, axis=-1)

  @nn.compact
  def __call__(self, query, key, value, attention_mask=None):
    b = query.shape[1]
    np = query.shape[2]
    sq = query.shape[0]
    sk = key.shape[0]
    hn = value.shape[3]

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query = jnp.reshape(query, (sq, b * np, hn))
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key = jnp.reshape(key, (sk, b * np, hn))

    norm_factor = jnp.sqrt(float(self.kv_channels)).astype(dtype)

    bmm1 = jnp.matmul(jnp.transpose(query, axes=(1, 0, 2)),
                      jnp.transpose(key, axes=(1, 2, 0))) / norm_factor

    # change view to [b, np, sq, sk]
    attention_scores = jnp.reshape(bmm1, newshape=(b, np, sq, sk))
    attention_probs = self.masked_softmax(
        attention_scores, attention_mask)

    attention_probs = nn.Dropout(
        rate=self.attention_dropout, deterministic=True)(attention_probs)

    # change view [sk, b * np, hn]
    value = jnp.reshape(value, newshape=(sk, b * np, hn))

    # change view [b * np, sq, sk]
    attention_probs = jnp.reshape(attention_probs, newshape=(b * np, sq, sk))

    # matmul: [b * np, sq, hn]
    context = jnp.matmul(attention_probs,
                         jnp.transpose(value, axes=(1, 0, 2)))

    # change view [b, np, sq, hn]
    context = jnp.reshape(context, newshape=(b, np, sq, hn))

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context = jnp.transpose(context, axes=(2, 0, 1, 3))

    # [sq, b, np, hn] --> [sq, b, hp]
    context = jnp.reshape(context, newshape=(sq, b, np * hn))

    return context


class BasicMLP(nn.Module):
  """Feed-forward network in Transformer layer"""
  hidden_size: int
  ffn_hidden_size: int
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    x = DenseLayer(self.ffn_hidden_size, kernel_init=self.kernel_init,
                   dtype=dtype, name='wi')(inputs)
    x = nn.gelu(x)

    output = DenseLayer(self.hidden_size, kernel_init=self.kernel_init,
                        dtype=dtype, name='wo')(x)
    return output

# ------------------------------------------------------------------------------
# Customized Layernorm - no subtraction of mean or bias.
# ------------------------------------------------------------------------------
class LayerNorm(nn.Module):
  """T5 Layer normalization operating on the last axis of the input data."""
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * jax.lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = param_with_axes(
        'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))

    scale = jnp.asarray(scale, self.dtype)
    return y * scale

class BasicTransformer(nn.Module):
  hidden_size: int
  ffn_hidden_size: int
  num_attention_heads: int
  layernorm_eps: float = 0.001
  attention_dropout: float = 0.01
  hidden_dropout: float = 0.01

  def setup(self):
    self.ln1 = LayerNorm(epsilon=self.layernorm_eps, dtype=dtype)
    self.qkv_projection = DenseLayer(3 * self.hidden_size, dtype=dtype)

    self.kv_channels = self.hidden_size // self.num_attention_heads
    self.attention = DotProductAttention(
        self.num_attention_heads,
        self.kv_channels,
        self.attention_dropout)

    self.projection = DenseLayer(self.hidden_size, dtype=dtype)
    self.dropout = nn.Dropout(self.hidden_dropout, deterministic=True)
    self.ln2 = LayerNorm(epsilon=self.layernorm_eps, dtype=dtype)
    self.mlp = BasicMLP(
        hidden_size=self.hidden_size,
        ffn_hidden_size=self.ffn_hidden_size,
    )

  def __call__(self, inputs, attention_mask=None):
    res = inputs
    x = self.ln1(inputs)

    # Fused QKV projection
    qkv = self.qkv_projection(x)
    qkv_shape = qkv.shape
    qkv = jnp.reshape(qkv,
                      newshape=(qkv_shape[0], qkv_shape[1],
                                self.num_attention_heads, 3 * self.kv_channels))
    q, k, v = jnp.split(qkv, 3, axis=-1)

    x = self.attention(q, k, v, attention_mask)
    x = self.projection(x)
    x = self.dropout(x)
    x = res + x
    res = x
    x = self.ln2(x)
    x = self.mlp(x)

    return x + res

# Layer configuration
hidden_size = 4096
sequence_length = 2048
batch_size = 4
ffn_hidden_size = 16384
num_attention_heads = 32

def run_benchmark():
  key = jax.random.PRNGKey(12)
  x_shape = (sequence_length, batch_size, hidden_size)
  x_data = jax.random.uniform(key, shape=x_shape, dtype=dtype)
  y_data = jax.random.uniform(key, shape=x_shape, dtype=dtype)
  
  timing_iters = 20
  warmup_iters = 20

  basic_transformer = BasicTransformer(
      hidden_size,
      ffn_hidden_size,
      num_attention_heads,
      attention_dropout=dropout_rate,
      hidden_dropout=dropout_rate,
  )

  initialized_var = basic_transformer.init(key, x_data)
  opt = optax.adam(learning_rate=0.1)
  ts_args = {'tx': opt, 'apply_fn': basic_transformer.apply}
  ts_args['model_variables' if use_fp8 else 'params'] = initialized_var
  state = TrainState.create(**ts_args)

  def step_fn(state, x, labels):
    def loss_fn(vars, x, labels):
      y = state.apply_fn(vars, x)
      loss = jnp.mean(jnp.square(y - labels))
      return loss

    grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1])
    loss, grads = grad_fn(state.model_variables if use_fp8 else state.params,
                          x_data, y_data)
    state = state.apply_gradients(grads=grads[0])
    return state, loss

  pjit_train_step_fn = pjit(step_fn)

  # Warmup runs
  for _ in range(warmup_iters):
    state, loss = pjit_train_step_fn(state, x_data, y_data)

  st = time.time()
  for _ in range(timing_iters):
    state, loss = pjit_train_step_fn(state, x_data, y_data)
  elapsed_time = (time.time() - st) / timing_iters * 1000
  print(f"Mean time: {elapsed_time} ms")

run_benchmark()
