# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.python.platform import tf_logging
from itertools import permutations

# A PoC optimizer wrapper to perform pruning with masks.
class AspOptimizerWrapper(optimizers.Optimizer):
  def __init__(self, optimizer, model, denylist=None, allowlist=None,
               padding=False, name=None, **kwargs):
    super(AspOptimizerWrapper, self).__init__(name, **kwargs)
    self._optimizer = optimizer
    self._padding = padding
    self._set_eligible_set(model, denylist=denylist, allowlist=allowlist)
    # This is a 6x4 matrix to store all possible 2:4 patterns.
    self._patterns = tf.convert_to_tensor(
                         list(set(permutations([0., 0., 1., 1.]))))

    self._set_hyper("learning_rate", optimizer.learning_rate)
    self._set_hyper("decay", optimizer.decay)

  def _prepare(self, var_list):
    return self._optimizer._prepare(var_list)

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list)
    for var in var_list:
      if var.ref() in self._eligible_set:
        self.add_slot(var, "mask")

  def _m4n2_1d(self, matrix):
    m, n = 4, 2

    mat = tf.math.abs(tf.reshape(matrix, shape=(-1, m)))
    pmax = tf.math.argmax(tf.linalg.matmul(mat, tf.transpose(self._patterns)),
                          axis=1)
    mask = tf.gather(self._patterns, pmax)
    mask = tf.reshape(mask, shape=matrix.shape)

    return mask

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):

     apply_gradients_op = super(AspOptimizerWrapper, self).apply_gradients(
         grads_and_vars, name, experimental_aggregate_gradients)

     # Normally self._optimizer.iterations is incremented in
     # self._optimizer.apply_gradients(). Since that is not called, we increment
     # it here instead.

     with tf.control_dependencies([apply_gradients_op]):
        return self._optimizer.iterations.assign_add(1)


  def _resource_apply_dense(self, grad, var, apply_state):
    if not var.ref() in self._eligible_set:
      return self._optimizer._resource_apply_dense(grad, var, apply_state)

    # The masks are only updated before the first step.
    mask = self.get_slot(var, "mask")

    def update_mask():
      # Conv2D stores a 4D filter weight and Dense stores a 2D kernel weight.
      # For Conv2D, the filter is in the shape of (H, W, I, O) and we need to
      # permute it to (H*W*O, I) and prune it along I. For Dense, the kernel is
      # in shape of (I, O) and we need to permute it to (O, I) and prune it
      # along I.
      if var.shape.rank == 2:
        matrix = tf.transpose(var, perm=[1, 0])
        orig_input_dim = matrix.shape[1]
        m = 4
        padding_size = m - orig_input_dim % m
        if self._padding and padding_size != 0:
          matrix = tf.pad(matrix, [[0, 0], [0, padding_size]], "CONSTANT")
      elif var.shape.rank == 4:
        matrix = tf.transpose(var, perm=[0, 1, 3, 2])
        permuted_shape = matrix.shape
        matrix = tf.reshape(matrix, shape=(-1, matrix.shape[-1]))

      new_mask = self._m4n2_1d(matrix)

      if var.shape.rank == 2:
        if self._padding and padding_size != 0:
          new_mask = new_mask[:, :orig_input_dim]
        new_mask = tf.transpose(new_mask, perm=[1, 0])
      elif var.shape.rank == 4:
        new_mask = tf.reshape(new_mask, shape=permuted_shape)
        new_mask = tf.transpose(new_mask, perm=[0, 1, 3, 2])

      update_mask_op = mask.assign(new_mask)
      with tf.control_dependencies([update_mask_op]):
        return tf.identity(mask)

    updated_mask = tf.cond(self._iterations == 0, update_mask,
                           lambda: tf.identity(mask))

    with tf.control_dependencies([updated_mask]):
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

  def get_config(self):
    return self._optimizer.get_config()

  # For both Dense and Conv2D, we need to make sure the output dim is a
  # multiple of 8 and the input dim is a multiple of 16 to use the
  # sparse tensor cores. For Dense layer, the kernel is a 2D matrix with
  # (I, O). For Conv2D layer, since the filter is in the format of HWIO,
  # the implicit GEMM will view it as a matrix of (H*W*I, O). In such
  # case, we simply apply a conservative restriction by requiring I be a
  # multiple of 16.
  def _check_valid_layer(self, layer):
    if (not isinstance(layer, layers.Dense) and
        not isinstance(layer, layers.Conv2D)):
      return False
    if layer.kernel.shape[-1] % 8 == 0:
      # Padding mode only supports padding the input dim in Dense layer.
      if self._padding:
        if isinstance(layer, layers.Dense):
          return True
        if (isinstance(layer, layers.Conv2D) and
            layer.kernel.shape[2] % 16 == 0):
          return True
      else:
        if (isinstance(layer, layers.Dense) and
            layer.kernel.shape[0] % 16 == 0):
          return True
        if (isinstance(layer, layers.Conv2D) and
            layer.kernel.shape[2] % 16 == 0):
          return True
    return False

  def _set_eligible_set(self, model, denylist, allowlist):
    if denylist and allowlist:
      raise ValueError("denylist and allowlist cannot be both defined.")
    if not denylist and not allowlist:
      allowlist = [layers.Dense, layers.Conv2D]

    target_list = allowlist if allowlist else denylist
    list_msg = "("
    for layer_def in target_list:
      list_msg += "%s, " % layer_def.__name__
    list_msg += ")"

    def layer_to_name(layer):
      type_name = "Unknown"
      if isinstance(layer, layers.Dense):
        type_name = "Dense"
      elif isinstance(layer, layers.Conv2D):
        type_name = "Conv2D"
      return type_name

    eligible_set = set()
    model_layers = model.submodules
    if allowlist:
      tf_logging.warn("[TF-ASP] Allowlist is used: %s" % list_msg)

      for layer in model_layers:
        if layer.__class__ in allowlist and self._check_valid_layer(layer):
          eligible_set.add(layer.kernel.ref())
          tf_logging.warn(
              "[TF-ASP] Pruning list accepts the \"kernel\" variable from "
              "layer: %s (type=%s, shape=%s)" % (layer.name,
                                                 layer_to_name(layer),
                                                 layer.kernel.shape))
    else:
      assert(denylist)
      tf_logging.warn("[TF-ASP] Denylist is used: %s" % list_msg)

      for layer in model_layers:
        if layer.__class__ not in denylist and self._check_valid_layer(layer):
          eligible_set.add(layer.kernel.ref())
          tf_logging.warn(
              "[TF-ASP] Pruning list accepts the \"kernel\" variable from "
              "layer: %s (type=%s, shape=%s)" % (layer.name,
                                                 layer_to_name(layer),
                                                 layer.kernel.shape))

    self._eligible_set = eligible_set

