import argparse
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import (Any, Callable, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import flax
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax import linen as nn
from flax import struct
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from tensorflow import summary

from fp8layers.jax import Dense

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--fp8', action='store_true', help='use_fp8')
parser.add_argument('--scale', type=int, help='model_scale', default=1)
args = parser.parse_args()

model_size_scale = args.scale
use_fp8 = args.fp8
print("DEBUG: use_fp8", use_fp8)
print("DEBUG: model_scale", model_size_scale)

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

Initializer = Callable[[PRNGKey, Shape, DType], Array]

def tree_shape(x): return jax.tree_map(lambda v: v.shape, x)

ext_kwargs = {}
if use_fp8:
  DenseLayer = Dense
  ext_kwargs['use_quant'] = True
else:
  DenseLayer = nn.DenseGeneral

def _convert_to_activation_function(
        fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  hidden_size: int = 2048
  ffn_hidden_size: int = 2048
  activations: Sequence[Union[str, Callable]] = ('gelu',)
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      x = DenseLayer(
          self.ffn_hidden_size,
          kernel_init=self.kernel_init,
          name=dense_name, **ext_kwargs)(inputs)
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    # Apply dropout and final dense output projection.
    output = DenseLayer(
        self.hidden_size,
        kernel_init=self.kernel_init,
        name='wo', **ext_kwargs)(x)
    return output

class DotProductAttention(nn.Module):
  """Attention operation in Transformer layer"""
  num_attention_heads: int
  kv_channels: int
  attention_dropout: float = 0.01

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
    b = query.shape[0]
    np = query.shape[2]
    sq = query.shape[1]
    sk = key.shape[1]
    hn = value.shape[3]

    # [b, sq, np, bn] -> [b, np, sq, bn]
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 3, 1))
    query = jnp.reshape(query, (b * np, sq, hn))
    key = jnp.reshape(key, (b * np, hn, sk))

    norm_factor = jnp.sqrt(float(self.kv_channels))
    bmm1 = jnp.matmul(query, key) / norm_factor

    # change view to [b, np, sq, sk]
    attention_scores = jnp.reshape(bmm1, (b, np, sq, sk))

    attention_probs = self.masked_softmax(
        attention_scores, None)  # attention_mask)

    attention_probs = nn.Dropout(
        rate=self.attention_dropout, deterministic=True)(attention_probs)

    attention_probs = self.masked_softmax(attention_scores, attention_mask)

    # change view [sk, b * np, hn]
    # value = jnp.reshape(value, (sk, b * np, hn))
    value = jnp.reshape(
        jnp.transpose(value, axes=(0, 2, 1, 3)),
        (b * np, sk, hn))

    # change view [b * np, sq, sk]
    attention_probs = jnp.reshape(attention_probs, (b * np, sq, sk))

    context = jnp.matmul(attention_probs, value)

    # change view to [b*np, sq, hn] - >[b, sq, np * hn]
    context = jnp.reshape(context, (b, np, sq, hn))
    context = jnp.transpose(context, axes=(0, 2, 1, 3))
    context = jnp.reshape(context, (b, sq, np * hn))
    return context

class BasicTransformer(nn.Module):
  use_quant: bool = False
  hidden_size: int = 512
  ffn_hidden_size: int = 1024
  num_attention_heads: int = 8
  layernorm_eps: float = 0.001
  attention_dropout: float = 0.01
  hidden_dropout: float = 0.01

  def setup(self):
    self.ln1 = nn.LayerNorm(epsilon=self.layernorm_eps)
    self.attention = DotProductAttention(
        num_attention_heads=self.num_attention_heads,
        kv_channels=self.hidden_size // self.num_attention_heads,
        attention_dropout=self.attention_dropout)
    self.dropout = nn.Dropout(self.hidden_dropout, deterministic=True)

    self.ln2 = nn.LayerNorm(epsilon=self.layernorm_eps)
    self.mlp = MlpBlock(hidden_size=self.hidden_size,
                        ffn_hidden_size=self.ffn_hidden_size,)
    self.projection = DenseLayer(
        self.hidden_size, **ext_kwargs)
    self.qkv_projection = DenseLayer(
        3 * self.hidden_size, **ext_kwargs)

  def __call__(self, inputs):
    res = inputs
    x = self.ln1(inputs)
    qkv = self.qkv_projection(x)
    qkv_shape = qkv.shape
    new_shape = tuple([qkv_shape[0], qkv_shape[1], self.num_attention_heads,
                      3 * self.hidden_size // self.num_attention_heads])
    qkv = jnp.reshape(qkv, new_shape)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    x = self.attention(q, k, v)
    x = self.projection(x)
    x = self.dropout(x)
    x = res + x
    res = x
    x = self.ln2(x)
    x = self.mlp(x)
    return x + res


class TrainState(struct.PyTreeNode):
  step: int
  params: Any
  grad_qscale_placeholder: Any
  qscale: Any
  opt_state: optax.OptState
  tx: optax.GradientTransformation = struct.field(pytree_node=False)

  @staticmethod
  def create(vars, tx):
    params = flax.core.unfreeze(vars['params'])
    opt_state = tx.init(params)
    grad_qscale_placeholder = flax.core.unfreeze(
        vars['grad_qscale_placeholder']) if 'grad_qscale_placeholder' in vars else None
    qscale = flax.core.unfreeze(vars['qscale']) if 'qscale' in vars else None
    return TrainState(0, params, grad_qscale_placeholder, qscale, opt_state, tx)

  def get_diff_vars(self):
    if self.grad_qscale_placeholder:
      return {'params': self.params, "grad_qscale_placeholder": self.grad_qscale_placeholder}
    return {'params': self.params}

  def get_nondiff_vars(self):
    if self.qscale:
      return {'qscale': self.qscale}
    return {}

def loss_fn(model, diff_vars, nondiff_vars, input_batch):
  logits, updated_nondiff_vars = model.apply(
      {**diff_vars, **nondiff_vars},
      input_batch['x'],
      mutable=['qscale'])
  batched_loss = optax.l2_loss(logits, input_batch['y'])
  return jnp.mean(batched_loss), updated_nondiff_vars

def step_fn(model, train_state, input_batch):
  bound_loss_fn = partial(loss_fn, model)
  grad_fn = jax.value_and_grad(bound_loss_fn, has_aux=True)
  (loss_val, updated_nondiff_vars), diff_vars_grads = grad_fn(
      train_state.get_diff_vars(), train_state.get_nondiff_vars(), input_batch)
  params_updates, updated_opt_state = train_state.tx.update(
      diff_vars_grads['params'], train_state.opt_state, params=train_state.params)
  updated_params = optax.apply_updates(train_state.params, params_updates)
  # Update train state
  new_qscale_vars = updated_nondiff_vars['qscale'] if 'qscale' in updated_nondiff_vars else None

  # Update qscale with grad_qscale_placeholder for gradient scale entries.
  if 'qscale' in updated_nondiff_vars:
    grad_qscale_vals = {
        tuple(re.sub(r'_placeholder$', '', '/'.join(k)).split('/')): v
        for k, v in flatten_dict(diff_vars_grads['grad_qscale_placeholder']).items()
    }
    flat_new_qscale_vars = flatten_dict(new_qscale_vars)
    flat_new_qscale_vars.update(grad_qscale_vals)
    new_qscale_vars = unflatten_dict(flat_new_qscale_vars)

  return train_state.replace(
      step=train_state.step + 1, params=updated_params, qscale=new_qscale_vars,
      opt_state=updated_opt_state), loss_val


batch_size = 4
epochs = 50

hidden_size = 4096 * model_size_scale
ffn_hidden_size = 16384 * model_size_scale
num_attention_heads = 32
sequence_length = 2048
dropout_rate = 0.0

kdata = jax.random.PRNGKey(123)
xkey, ykey, xekey, yekey = jax.random.split(kdata, 4)
x_train = jax.random.uniform(xkey, shape=(
    batch_size, sequence_length, hidden_size))
x_eval = jax.random.uniform(ykey, shape=(
    batch_size // 2, sequence_length, hidden_size))
y_train = jax.random.uniform(xekey, shape=(
    batch_size, sequence_length, hidden_size))
y_eval = jax.random.uniform(yekey, shape=(
    batch_size // 2, sequence_length, hidden_size))
train_size = x_train.shape[0]


LOG_DIR = './model_3'
def run(use_quant: bool, tb_label: str):
  root_k = jax.random.PRNGKey(123)
  init_k, subk = jax.random.split(root_k)
  model = BasicTransformer(
      use_quant=use_quant, hidden_size=hidden_size,
      ffn_hidden_size=ffn_hidden_size, num_attention_heads=num_attention_heads,
      attention_dropout=dropout_rate, hidden_dropout=dropout_rate)
  init_vars = model.init(init_k, x_train)
  tx = optax.sgd(learning_rate=0.01, momentum=0.1)
  train_state = TrainState.create(init_vars, tx)
  summary_writer = tf.summary.create_file_writer('%s/%s' % (LOG_DIR, tb_label))

  train_step = jax.jit(partial(step_fn, model))
  eval_loss_fn = jax.jit(partial(loss_fn, model))
  step = 0
  for epoch_i in range(epochs):
    num_steps = train_size // batch_size
    for i in range(num_steps):
      input_batch = {
          'x': x_train[i * batch_size: (i + 1) * batch_size],
          'y': y_train[i * batch_size: (i + 1) * batch_size]}
      train_state, train_loss = train_step(train_state, input_batch)
      # For debugging only, otherwise it slows down training
      with summary_writer.as_default(step=step):
        if train_state.qscale:
          # print(f'epoch={epoch_i}, step={i}, train_state.qscale={train_state.qscale}')
          # Monitor quantization scales
          for k, v in flatten_dict(train_state.qscale).items():
            tf.summary.scalar('/'.join(k), v)
        tf.summary.scalar('train_loss', train_loss)
      step += 1
    eval_input_batch = {'x': x_eval, 'y': y_eval}
    eval_loss, _ = eval_loss_fn(
        train_state.get_diff_vars(),
        train_state.get_nondiff_vars(),
        input_batch)
    with summary_writer.as_default(step=step):
      tf.summary.scalar('eval_loss', eval_loss)
    print(
        f'epoch={epoch_i}, step={i}, train_loss={train_loss}, eval_loss={eval_loss}')

st = time.time()
run(use_quant=use_fp8, tb_label='fp8')
print(time.time() - st)
