from typing import Callable, Optional, Dict, Any, Tuple

import optax

from flax import core
from flax.core import scope as flax_scope
from flax import linen as nn
from flax import traverse_util
from flax import struct
from flax.linen import partitioning as nn_partitioning
from jax import lax


# Type annotations
EMPTY_DICT = core.freeze({})
FrozenDict = flax_scope.FrozenDict
FrozenVariableDict = flax_scope.FrozenVariableDict
VariableDict = flax_scope.VariableDict

def _validate_params_axes(params_axes, params):
  axis_names = nn_partitioning.get_axis_names(params_axes)
  missing_params_axes = (
      set(traverse_util.flatten_dict(params, sep='/')) -
      set(traverse_util.flatten_dict(axis_names, sep='/')))
  if missing_params_axes:
    raise ValueError(
        f'Missing axis names for parameters: {missing_params_axes}')

def _split_variables_and_axes(
    variables_and_axes: FrozenVariableDict
) -> Tuple[FrozenVariableDict, FrozenVariableDict]:
  """Splits `variables_and_axes` into two separate dicts with the same keys."""
  # For each `key`, `key_axes` (if any) are its axes in `variables_and_axes`.
  variables = {}
  axes = {}
  for k, v in variables_and_axes.items():
    if k.endswith('_axes'):
      axes[k[:-5]] = v  # k without "_axes".
      _validate_params_axes(v, variables_and_axes[k[:-5]])  # k without "_axes".
    else:
      variables[k] = v
  return core.freeze(variables), core.freeze(axes)

class TrainState(struct.PyTreeNode):
  """
  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
      convenience to have a shorter params list for the `train_step()` function
      in your training loop.
    model_variables: The params that needs to be updated.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
  """
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  model_variables: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads, **kwargs):
    # For the variables in the params collection, we will use the optimizer as
    # usual.
    updates, new_opt_state = self.tx.update(
        grads['params'], self.opt_state, self.model_variables['params'])
    new_params = optax.apply_updates(self.model_variables['params'], updates)

    update_model_variables = core.unfreeze(self.model_variables)
    update_model_variables['params'] = new_params

    # For the fp8 variables in the fp8-params collection, we will simply replace
    # them with their grads, because their grads are actually new values defined
    # in the custom_vjp functions.
    if 'fp8_params' in grads:
      update_model_variables['fp8_params'] = grads['fp8_params']

    return self.replace(
        step=self.step + 1,
        model_variables=core.freeze(update_model_variables),
        opt_state=new_opt_state,
    )

  @classmethod
  def create(cls, apply_fn, model_variables, tx):
    """Creates a new instance with `step=0` and initialized `opt_state`."""

    # For some unknown reason, we have to explicitly freeze the model variables
    # to make sure the initialized optimizer states are also frozen. Otherwise,
    # the runtime would complain about the unexpected dtype.
    model_variables = core.freeze(model_variables)

    params = model_variables['params']

    # We assume all the params are annotated when the params_axes is specified.
    if 'params_axes' in model_variables:
      params_axes = model_variables['params_axes']
      _validate_params_axes(params_axes, params)

    opt_state = tx.init(params)

    if 'fp8_params' in model_variables:
      fp8_params = model_variables['fp8_params']
      fp8_params_axes = model_variables['fp8_params_axes']
      _validate_params_axes(fp8_params_axes, fp8_params)

    return cls(
        step=0,
        apply_fn=apply_fn,
        model_variables=model_variables,
        tx=tx,
        opt_state=opt_state,
    )
