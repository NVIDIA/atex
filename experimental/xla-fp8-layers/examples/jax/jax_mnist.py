import argparse
import re
from functools import partial
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
from fp8layers.jax import Dense

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--fp8', action='store_true', help='use_fp8')
args = parser.parse_args()
use_fp8 = args.fp8
print("DEBUG: use_fp8", use_fp8)

if use_fp8:
  DenseLayer = Dense
else:
  class DenseLayer(nn.Dense):
    features: int
    activation: Optional[Callable] = None

    @nn.compact
    def __call__(self, inputs):
      y = nn.Dense(self.features, use_bias=self.use_bias)(inputs)
      if self.activation is not None:
        y = self.activation(y)
      return y


class MnistModel(nn.Module):
  use_quant: bool = False
  def setup(self):
    self.dense1 = DenseLayer(64, activation=jax.nn.relu)
    self.dense2 = DenseLayer(64, activation=jax.nn.relu)
    # Just to meet multiple of 16 requirement of cublasLt
    self.dense3 = DenseLayer(16)

  def __call__(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    output = self.dense3(x)
    return output[:,:10]

import tensorflow as tf
from flax import struct
import optax
from dataclasses import dataclass

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
    grad_qscale_placeholder=flax.core.unfreeze(vars['grad_qscale_placeholder']) if 'grad_qscale_placeholder' in vars else None
    qscale=flax.core.unfreeze(vars['qscale']) if 'qscale' in vars else None
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
  logits, updated_nondiff_vars = model.apply({**diff_vars, **nondiff_vars}, input_batch['x'], mutable=['qscale'])
  batched_loss = optax.softmax_cross_entropy_with_integer_labels(logits, input_batch['y'])
  return jnp.mean(batched_loss, axis=0), updated_nondiff_vars

def step_fn(model, train_state, input_batch):
  bound_loss_fn = partial(loss_fn, model)
  grad_fn = jax.value_and_grad(bound_loss_fn, has_aux=True)
  (loss_val, updated_nondiff_vars), diff_vars_grads = grad_fn(train_state.get_diff_vars(), train_state.get_nondiff_vars(), input_batch)
  params_updates, updated_opt_state = train_state.tx.update(diff_vars_grads['params'], train_state.opt_state, params=train_state.params)
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

  return train_state.replace(step=train_state.step+1, params=updated_params, qscale=new_qscale_vars, opt_state=updated_opt_state), loss_val
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

validation_split = 0.2

validation_size = int(60000*0.2)
x_eval = x_train[-validation_size:]
y_eval = y_train[-validation_size:]
x_train = x_train[0:-validation_size]
y_train = y_train[0:-validation_size]
train_size = x_train.shape[0]

batch_size = 64
epochs = 60

from tensorflow import summary

LOG_DIR='./model_3'
def run(use_quant: bool, tb_label: str):
  root_k = jax.random.PRNGKey(123)
  init_k, subk = jax.random.split(root_k)
  model = MnistModel(use_quant=use_quant)
  init_vars = model.init(init_k, x_eval)
  tx = optax.sgd(learning_rate=0.01, momentum=0.1)
  train_state = TrainState.create(init_vars, tx)
  summary_writer = tf.summary.create_file_writer('%s/%s' %(LOG_DIR, tb_label))

  train_step = jax.jit(partial(step_fn, model))
  eval_loss_fn = jax.jit(partial(loss_fn, model))
  step = 0
  for epoch_i in range(epochs):
    # TODO: do not truncate?
    num_steps = train_size // batch_size
    for i in range(num_steps):
      input_batch = {'x': x_train[i * batch_size: (i + 1) * batch_size], 'y': y_train[i * batch_size: (i + 1) * batch_size]}
      train_state, train_loss = train_step(train_state, input_batch)
      # For debugging only, otherwise it slows down training
      with summary_writer.as_default(step=step):
#        if train_state.qscale:
#          print(f'epoch={epoch_i}, step={i}, train_state.qscale={train_state.qscale}')
          # Monitor quantization scales
#          for k, v in flatten_dict(train_state.qscale).items():
#            tf.summary.scalar('/'.join(k), v)
        tf.summary.scalar('train_loss', train_loss)
      step += 1
    eval_input_batch = {'x': x_eval, 'y': y_eval}
    eval_loss, _ = eval_loss_fn(train_state.get_diff_vars(), train_state.get_nondiff_vars(), input_batch)
    with summary_writer.as_default(step=step):
      tf.summary.scalar('eval_loss', eval_loss)
    print(f'epoch={epoch_i}, step={i}, train_loss={train_loss}, eval_loss={eval_loss}')

import time
st = time.time()
run(use_quant=True, tb_label='fp8')
print('sec:', time.time()-st)
