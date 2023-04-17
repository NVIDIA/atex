# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import numpy as np
import tensorflow as tf

from itertools import permutations
from tensorflow.keras import layers, optimizers
from tensorflow.python.platform import tf_logging

from .permuting_utils import permute_model
from .pruning_utils import get_2to4_mask
from .tf_asp_logging import SHOW_PRUNING_INFO

PRUNABLE_2x4_ALLOWLIST = (layers.Conv2D, layers.Dense)


def is_prunable(layer, padding):
  """Returns `True` if the `kernel` is prunable."""
  if type(layer) not in PRUNABLE_2x4_ALLOWLIST:
    return False

  kernel = layer.kernel
  # Check the output dim.
  if kernel.shape[-1] % 8 != 0:
    return False

  # Check the input dim.
  if type(layer) == layers.Dense:
    return kernel.shape[0] % 16 == 0 if not padding else True
  if type(layer) == layers.Conv2D:
    return kernel.shape[2] % 16 == 0

  return False


def find_prunable_kernels(model, padding):
  """Returns a set of variable refs that are prunable. """
  if model is None:
    raise ValueError('`model` cannot be None')

  if not isinstance(model, tf.keras.Model):
    raise ValueError('`model` can only be a `tf.keras.Model` instance.'
                     'You passed an instance of type: {input}.'.format(
                         input=model.__class__.__name__))
  if not model.built:
    raise ValueError('`model` must be a built model. '
                     'been built yet. Please call `model.build(input_shape)` '
                       'before pruning your model.')

  prunable_kernels = set()
  for layer in model.submodules:
    if is_prunable(layer, padding):
      prunable_kernels.add(layer.kernel.ref())
  return prunable_kernels


class AspOptimizerWrapperV2(optimizers.legacy.Optimizer):
  """An optimizer that automatically applies sparsity to the weights.

  `AspOptimizerWrapperV2` wraps another optimizer and applies the weight
  permutation (if necessary) and weight pruning.

  A typical usage:

  >>> import tf_asp
  >>> opt = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=1.0)
  >>> opt = tf_asp.AspOptimizerWrapperV2(opt, model, padding=True)

  Args:
    optimizer: The `tf.keras.optimizers.Optimizer` instance to wrap.
    model: The built model corresponds to the optimizer.
    input_shapes: A tuple or a list of tuples representing the input tensor
      shapes or TensorSpecs. This is required only when the input signature
      cannot be deduced from the model.
    padding: A boolean indicating whether padding is applied.
    permute: A boolean indicating whether the permutation is on. It is true by
      default.
    search_device: A string indicating which device the permutation searching
      uses: 'GPU' (default), 'CPU'.
    plot_to_file: (str or None) The path to save the op graph plot using pydot
      (if any). It is None by default.
  """
  def __init__(self, optimizer, model, input_shapes=None, padding=False,
               permute=True, search_device='GPU', search_time_limit=60,
               plot_to_file=None, name=None, **kwargs):
    super(AspOptimizerWrapperV2, self).__init__(name, **kwargs)

    self._optimizer = optimizer
    self._padding = padding

    self._prunable_kernels = find_prunable_kernels(model, padding)

    if permute:
      permute_model(model, input_shapes, self._prunable_kernels, search_device,
                    search_time_limit, plot_to_file)

    # A 6x4 matrix to store all combinations of 2:4 patterns. Allocate this
    # tensor inside the optimizer to avoid GPU allocation when importing. 
    self._patterns = tf.convert_to_tensor(
                            list(set(permutations([0., 0., 1., 1.]))))

    # We store a copy of learning_rate in _hyper, since the _hyper may be
    # directly accessed in some circumstances.
    # TODO(kaixih): check if we can remove such copy.
    self._hyper['learning_rate'] = optimizer.learning_rate

  def _prepare(self, var_list):
    return self._optimizer._prepare(var_list)

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list)
    for var in var_list:
      if var.ref() in self._prunable_kernels:
        self.add_slot(var, "mask")

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):

     apply_gradients_op = super(AspOptimizerWrapperV2, self).apply_gradients(
         grads_and_vars, name, experimental_aggregate_gradients)

     # Normally self._optimizer.iterations is incremented in
     # self._optimizer.apply_gradients(). Since that is not called, we increment
     # it here instead.

     with tf.control_dependencies([apply_gradients_op]):
        return self._optimizer.iterations.assign_add(1)

  def _resource_apply_dense(self, grad, var, apply_state):
    if not var.ref() in self._prunable_kernels:
      return self._optimizer._resource_apply_dense(grad, var, apply_state)

    mask = self.get_slot(var, "mask")

    def update_mask():
      new_mask = get_2to4_mask(var, self._padding, self._patterns)
      update_mask_op = mask.assign(new_mask)
      with tf.control_dependencies([update_mask_op]):
        return tf.identity(mask)

    # The masks are updated at the beginning of fine-tuning.
    maybe_update_mask = tf.cond(self._iterations == 0, update_mask,
                                lambda: tf.identity(mask))

    with tf.control_dependencies([maybe_update_mask]):
      opt_op = self._optimizer._resource_apply_dense(grad, var, apply_state)
      with tf.control_dependencies([opt_op]):
        new_var = tf.math.multiply(var, mask)
        return var.assign(new_var)

  def _resource_apply_sparse(self, grad, var, indices, apply_state):
    return self._optimizer._resource_apply_sparse(grad, var, indices,
                                                  apply_state)

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices,
                                               **kwargs):
    return self._optimizer._resource_apply_sparse_duplicate_indices(
               grad, handle, indices, **kwargs)


  def __getattribute__(self, name):
    try:
      return object.__getattribute__(self, name)
    except AttributeError as e:
      if name == '_optimizer' or name == '_hyper':
        # Avoid infinite recursion
        raise e

      # Delegate hyperparameter accesses to inner optimizer.
      if name == 'lr':
        name = 'learning_rate'
      if name in self._optimizer._hyper:
        return self._optimizer._get_hyper(name)
      raise e


  def __dir__(self):
    result = set(super(AspOptimizerWrapperV2, self).__dir__())
    if '_optimizer' in result:
      result |= self._optimizer._hyper.keys()
      if 'learning_rate' in self._optimizer._hyper.keys():
        result.add('lr')
    return list(result)


  def __setattr__(self, name, value):
    if name == 'lr':
      name = 'learning_rate'
    # Delegate setting hyperparameter to inner optimizer if the attribute does
    # not exist on the AspOptimizerWrapperV2 
    try:
      # We cannot check for the 'iterations' attribute as it cannot be set after
      # it is accessed.
      if name != 'iterations':
        object.__getattribute__(self, name)
      has_attribute = True
    except AttributeError:
      has_attribute = False
    if (name != '_optimizer' and hasattr(self, '_optimizer') and
        name in self._optimizer._hyper and not has_attribute):
      self._optimizer._set_hyper(name, value)
      # We need to update the wrapper's _hyper, since we store a copy of
      # learning_rate.
      if name == "learning_rate":
        self._set_hyper(name, value)
    else:
      super(AspOptimizerWrapperV2, self).__setattr__(name, value)


  def get_config(self):
    serialized_optimizer = tf.keras.optimizers.serialize(self._optimizer)
    return {
        'optimizer': serialized_optimizer,
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return tf.keras.optimizers.deserialize(
        config['optimizer'], custom_objects=custom_objects)

def check_pruned_layers(model, show_all=False, check_structured=False):
  """Checks how many layers are pruned. """
  model_layers = model.submodules
  layers_count = 0
  pruned_count = 0
  for layer in model_layers:
    if type(layer) in (layers.Conv2D, layers.Dense):
      layers_count += 1
      total_count = tf.size(layer.kernel)
      nonzero_count = tf.math.count_nonzero(layer.kernel)
      nonzero_count = tf.cast(nonzero_count, 'int32')
      zero_ratio = (total_count - nonzero_count) / total_count
      if abs(zero_ratio - 0.5) < 0.003:
        pruned_count += 1
      is_structured_str = ""
      if check_structured:
        is_structured = True
        weights = layer.kernel.numpy()
        if type(layer) is layers.Conv2D:
          K = layer.kernel.shape[3]
          C = layer.kernel.shape[2]
          R = layer.kernel.shape[0]
          S = layer.kernel.shape[1]
          for k in range(K):
            for r in range(R):
              for s in range(S):
                for c_packed in range(0, C // 4):
                  if np.count_nonzero(weights[r, s, c_packed*4:(c_packed+1)*4, k]) > 2:
                    is_structured = False
        if type(layer) is layers.Dense:
          K = layer.kernel.shape[1]
          C = layer.kernel.shape[0]
          for k in range(K):
            for c_packed in range(0, C // 4):
              if np.count_nonzero(weights[c_packed*4:min((c_packed+1)*4,C), k]) > 2:
                is_structured = False
        is_structured_str = "Structured=" + str(is_structured)

      if show_all:
        print("[TF-ASP] layer=%s, type=%s, shape=%s: zero_ratio=%f %s" % (
            layer.name, type(layer).__name__, layer.kernel.shape, zero_ratio, is_structured_str))
  print("[TF-ASP] %d/%d layers (Conv2D or Dense) are pruned!" % (pruned_count,
                                                                 layers_count))
  return pruned_count, layers_count

