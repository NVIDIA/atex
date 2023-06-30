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
  flax_mutables: FrozenDict = EMPTY_DICT
  validate_axes: bool = True
  # Contains axis metadata (e.g., names) matching flax_mutables tree.
  flax_mutables_axes: Optional[FrozenVariableDict] = None

  def variables(self) -> core.FrozenDict[str, Any]:
    return core.freeze({'params': self.params})
  
  def mutable_variables(self) -> core.FrozenDict[str, Any]:
    if self.flax_mutables:
      return core.freeze(self.flax_mutables)
    return core.freeze({})

  def apply_gradients(self, *, grads, flax_mutables=EMPTY_DICT, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads['params'], self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        flax_mutables=flax_mutables,
        opt_state=new_opt_state,
    )

  @classmethod
  def create(cls, apply_fn, model_variables, tx, validate_axes=True):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    other_variables, params = core.pop(model_variables, 'params')

    if 'params_axes' in other_variables:
      other_variables, params_axes = core.pop(
          other_variables, 'params_axes'
      )
      if validate_axes:
        _validate_params_axes(params_axes, params)
    else:
      params_axes = None

    # Split other_variables into mutables and their corresponding axes.
    flax_mutables, flax_mutables_axes = _split_variables_and_axes(
        other_variables
    )
    flax_mutables_axes = flax_mutables_axes or None
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        params_axes=params_axes,
        flax_mutables=flax_mutables,
        flax_mutables_axes=flax_mutables_axes,
        validate_axes=validate_axes,
    )
