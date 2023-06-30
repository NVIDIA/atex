"""Runs a simple mnist model with FP8. """
import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from flax import linen as nn
from flax.training import train_state
from fp8layers.flax import DenseGeneral, TrainState

parser = argparse.ArgumentParser(description='MNIST')
parser.add_argument('--fp8', action='store_true', help='use_fp8')
parser.add_argument('--mixed', action='store_true',
                    help='Enable mixed precision and bf16 compute type')
args = parser.parse_args()

use_fp8 = args.fp8
use_mixed = args.mixed

TrainState = TrainState if use_fp8 else train_state.TrainState
DenseLayer = DenseGeneral if use_fp8 else nn.DenseGeneral
dtype = jnp.bfloat16 if use_mixed else jnp.float32

class MnistModel(nn.Module):
  def setup(self):
    self.dense1 = DenseLayer(64, dtype=dtype)
    self.dense2 = DenseLayer(64, dtype=dtype)
    self.dense3 = DenseLayer(10, dtype=dtype)

  def __call__(self, inputs):
    x = self.dense1(inputs)
    x = nn.relu(x)
    x = self.dense2(x)
    x = nn.relu(x)
    output = self.dense3(x)
    return output

def step_fn(train_state, inputs, training):
  def loss_fn(diff_vars, x, labels, mutable_variables=None):
    if use_fp8:
      logits, _ = train_state.apply_fn({**diff_vars, **mutable_variables}, x,
                                       mutable=['fp8_params'])
    else:
      logits = train_state.apply_fn(diff_vars, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss, axis=0)

  if training:
    if use_fp8:
      grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 3])
      loss_val, vars_grads = grad_fn(train_state.variables(), inputs['x'],
                                     inputs['y'],
                                     train_state.mutable_variables())
      new_state = train_state.apply_gradients(grads=vars_grads[0],
                                              flax_mutables=vars_grads[1])
    else:
      grad_fn = jax.value_and_grad(loss_fn, argnums=[0])
      loss_val, vars_grads = grad_fn(train_state.params, inputs['x'],
                                     inputs['y'])
      new_state = train_state.apply_gradients(grads=vars_grads[0])
    return new_state, loss_val
  else:
    if use_fp8:
      input_args = {'diff_vars': train_state.variables(), 'x': inputs['x'], 
                    'labels': inputs['y'],
                    'mutable_variables': train_state.mutable_variables()}
    else:      
      input_args = {'diff_vars': train_state.params, 'x': inputs['x'],
                    'labels': inputs['y']}
    loss_val = loss_fn(**input_args)
    return loss_val

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

validation_size = int(60000 * 0.2)
x_train = x_train[0:-validation_size]
y_train = y_train[0:-validation_size]
x_eval = x_train[-validation_size:]
y_eval = y_train[-validation_size:]
train_size = x_train.shape[0]

batch_size = 64
epochs = 20

def run():
  root_key = jax.random.PRNGKey(123)
  init_key, _ = jax.random.split(root_key)

  model = MnistModel()
  variables = model.init(init_key, x_train)

  tx = optax.adam(learning_rate=0.001)

  if use_fp8:
    train_state = TrainState.create(
        model_variables=variables, tx=tx, apply_fn=model.apply)
  else:
    train_state = TrainState.create(
        params=variables, tx=tx, apply_fn=model.apply)

  train_step_fn = jax.jit(partial(step_fn, training=True))
  eval_step_fn = jax.jit(partial(step_fn, training=False))

  for epoch_i in range(epochs):
    num_steps = train_size // batch_size
    for i in range(num_steps):
      train_batch = {'x': x_train[i * batch_size: (i + 1) * batch_size],
                     'y': y_train[i * batch_size: (i + 1) * batch_size]}
      train_state, train_loss = train_step_fn(train_state, train_batch)

    eval_batch = {'x': x_eval, 'y': y_eval}
    eval_loss = eval_step_fn(train_state, eval_batch)
    print(f'Epoch={epoch_i}, train_loss={train_loss}, eval_loss={eval_loss}')

st = time.time()
jax.block_until_ready(run())
print(f"Elapsed time: {time.time() - st} ms")
