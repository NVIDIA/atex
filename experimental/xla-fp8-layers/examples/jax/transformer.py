import argparse
import time

from functools import partial
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import optax

from flax import linen as nn
from flax.training import train_state
from fp8layers.jax import DenseGeneral

dropout_rate = 0.0

parser = argparse.ArgumentParser(description='Benchmark a basic encoder layer')
parser.add_argument('--fp8', action='store_true', help='Enable fp8')
parser.add_argument('--mixed', action='store_true',
                    help='Enable mixed precision and bf16 compute type')
args = parser.parse_args()

use_fp8 = args.fp8
use_mixed = args.mixed

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


class BasicTransformer(nn.Module):
  hidden_size: int
  ffn_hidden_size: int
  num_attention_heads: int
  layernorm_eps: float = 0.001
  attention_dropout: float = 0.01
  hidden_dropout: float = 0.01

  def setup(self):
    self.ln1 = nn.LayerNorm(epsilon=self.layernorm_eps)
    self.qkv_projection = DenseLayer(3 * self.hidden_size, dtype=dtype)

    self.kv_channels = self.hidden_size // self.num_attention_heads
    self.attention = DotProductAttention(
        self.num_attention_heads,
        self.kv_channels,
        self.attention_dropout)

    self.projection = DenseLayer(self.hidden_size, dtype=dtype)
    self.dropout = nn.Dropout(self.hidden_dropout, deterministic=True)
    self.ln2 = nn.LayerNorm(epsilon=self.layernorm_eps)
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

key = jax.random.PRNGKey(12)
x_shape = (sequence_length, batch_size, hidden_size)
x_data = jax.random.uniform(key, shape=x_shape, dtype=dtype)
y_data = jax.random.uniform(key, shape=x_shape, dtype=dtype)

basic_transformer = BasicTransformer(
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    attention_dropout=dropout_rate,
    hidden_dropout=dropout_rate,
)

init_vars = basic_transformer.init(key, x_data)

def loss_fn(variables, x, labels):
  y = basic_transformer.apply(variables, x)
  loss = jnp.mean(jnp.square(y - labels))
  return loss

@jax.jit
def run_training(variables, x, labels):
  grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1])
  y, grads = grad_fn(variables, x, labels)
  return y, grads[1], grads[0]

timing_iters = 20
warmup_iters = 20

def run_benchmark(iters, x, y):
  for _ in range(iters):
    out, dx, _ = run_training(init_vars, x, y)
  return out, dx

# Warmup runs
jax.block_until_ready(run_benchmark(warmup_iters, x_data, y_data))

# Timing runs
st = time.time()
jax.block_until_ready(run_benchmark(timing_iters, x_data, y_data))
elapsed_time = (time.time() - st) / timing_iters * 1000
print(f"Mean time: {elapsed_time} ms")

