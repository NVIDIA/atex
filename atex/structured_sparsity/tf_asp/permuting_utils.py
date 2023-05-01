# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import json
import numpy as np
import os
import pprint
import shutil
import tempfile
import tensorflow as tf
import time

from google.protobuf import json_format
from itertools import count
from tensorflow.keras import layers, optimizers, models
from tensorflow.python.platform import tf_logging

from .tf_asp_logging import *
from .permuting_search import Exhaustive_Search
from .permuting_search_utils import sum_after_2_to_4
from .permuting_search_utils import try_swap

try:
  # pydot-ng is a fork of pydot that is better maintained.
  import pydot_ng as pydot
except ImportError:
  # pydotplus is an improved version of pydot
  try:
    import pydotplus as pydot
  except ImportError:
    # Fall back on pydot if necessary.
    try:
      import pydot
    except ImportError:
      pydot = None

# To better preserve high magnitude weights, we permute weights before pruning.
# The permutation applied on the current op usually requires corresponding
# permutation on its upstream ops. Here we maintain four op lists to
# differentiate how the permutation is applied:
# * allowlist: GEMM-based ops that can benefit from sparse tensor cores.
# * inferlist: ops that require weight permutation when its downstream ops get
#              permuted.
# * clearlist: ops that won't be affected by the permutation and the permutation
#              sequences should be passed through to their upstream ops.
# * skiplist: this list is mainly used in the plot_to_file() to make the graph
#             more concise.
# All other ops are viewed as unsupported ops and the permutation should be
# stopped when they are found. A typical supported pattern is: 
# [ALLOWLIST] -> (INFERLIST|CLEARLIST)* -> [ALLOWLIST],
# where ()* means an arbitrary number of ops.
PERMUTABLE_2x4_ALLOWLIST = ('Conv2D', 'MatMul')
PERMUTABLE_2x4_INFERLIST = ('BiasAdd',
                            'FusedBatchNormV3')
PERMUTABLE_2x4_CLEARLIST = ('AddV2',
                            'Cast',
                            'Erf',
                            'Identity',
                            'Mul',
                            'RealDiv',
                            'Relu',
                            'Relu6',
                            'Rsqrt',
                            'Softmax',
                            'StopGradient',
                            'Sub')
PERMUTABLE_2x4_SKIPLIST = ('Const',
                           'Placeholder',
                           'ReadVariableOp')


def get_graph_def(model, input_shapes):
  """Gets the op graph def for the given model. """
  if (getattr(model, '_build_input_shape', None) is None and
      input_shapes == None):
    model_path = tempfile.mkdtemp()
    tf_logging.vlog(SHOW_PRUNING_INFO,
        "[TF-ASP] tmpdir is created: %s. This is used to store a temp "
        "savedmodel to extract input signatures. Users may specify "
        "`input_shapes` to skip this step.\n" % model_path)
    try:
      tf.saved_model.save(model, model_path)
      loaded = tf.saved_model.load(model_path)
      infer = loaded.signatures['serving_default']
      assert len(infer.structured_input_signature) == 2
      infer_inputs = infer.structured_input_signature[1]

      input_specs = []
      # Manually extract each input specs to make sure the order is correct.
      for i in range(len(infer_inputs)):
        input_name = 'input_' + str(i+1)
        input_specs.append(infer_inputs[input_name])
      if len(input_specs) != len(infer_inputs):
        raise ValueError
      tf_logging.vlog(SHOW_PRUNING_INFO,
          "[TF-ASP] Successfully found the input signature: [{}]\n".format(
              ', '.join([str(x) for x in input_specs])))
    except:
      raise ValueError("Failed to extract the input_shapes from the model. "
                       "Users may need to try manually specify 'input_shapes'")
    finally:
      tf_logging.vlog(SHOW_PRUNING_INFO,
                      "[TF-ASP] tmpdir is cleaned up: %s\n" % model_path)
      shutil.rmtree(model_path)
  else:
    if getattr(model, '_build_input_shape', None) is not None:
      model_input_shapes = model._build_input_shape
    else:
      model_input_shapes = input_shapes
    input_specs = []
    if not isinstance(model_input_shapes, list):
      if isinstance(model_input_shapes, tf.TensorSpec):
        input_specs.append(model_input_shapes)
      else:
        input_specs.append(tf.TensorSpec(shape=model_input_shapes))
    else:
      for input_shape in model_input_shapes:
        if isinstance(input_shape, tf.TensorSpec):
          input_specs.append(input_shape)
        else:
          input_specs.append(tf.TensorSpec(shape=input_shape))

  # For some custom models, the list is only expected to contain multiple
  # inputs.
  if len(input_specs) == 1:
    input_specs = input_specs[0]

  tf_fn = tf.function(lambda x: model(x))
  graph_def = tf_fn.get_concrete_function(input_specs).graph.as_graph_def()
  json_string = json_format.MessageToJson(graph_def)
  obj = json.loads(json_string)
  return obj


def build_layer_dict(model):
  """Builds a dict holds {layer_name: layer} from the flattened model. """
  def _is_module(obj):
    return isinstance(obj, tf.Module)
  layer_dict = {}
  if not hasattr(model, '_flatten'):
    return layer_dict
  layer_with_path = tuple(model._flatten(predicate=_is_module, with_path=True))
  for path, layer in layer_with_path:
    layer_name = model.name
    current_obj = model
    for subpath in path:
      if isinstance(subpath, str):
        if type(current_obj) is dict:
          current_obj = current_obj[subpath]
        else:
          current_obj = getattr(current_obj, subpath)
      else:
        assert isinstance(subpath, int)
        current_obj = current_obj[subpath]
      if _is_module(current_obj):
        layer_name += '/' + current_obj.name
    if layer_name not in layer_dict:
      layer_dict[layer_name] = [layer]
    elif layer not in layer_dict[layer_name]:
      layer_dict[layer_name].append(layer)

  return layer_dict


def find_variable(node_name, node_op, var_name, layer_dict):
  """Finds the variable for node_name. """
  if node_op not in PERMUTABLE_2x4_ALLOWLIST + PERMUTABLE_2x4_INFERLIST:
    return None

  # We assume the last subdir is the op name. So, we strip it to obtain the
  # prefix. For the MatMul, it is possible that the op name is
  # "Tensordot/MatMul", e.g., when the input tensor is not in 2D.
  prefix = node_name.rsplit('/', 1)[0]
  if node_op == 'MatMul' and prefix.endswith("Tensordot"):
    prefix = prefix.rsplit('/', 1)[0]

  # Try to find the variable from layer_dict.
  if prefix in layer_dict:
    layers = layer_dict[prefix]
    if len(layers) == 1:
      layer = layers[0]
      if var_name == 'kernel' and hasattr(layer, 'kernel'):
        return layer.kernel
      if var_name == 'bias' and hasattr(layer, 'bias'):
        return layer.bias
      if var_name == 'gamma' and hasattr(layer, 'variables'):
        return layer.variables[0]
      if var_name == 'beta' and hasattr(layer, 'variables'):
        return layer.variables[1]
      if var_name == 'moving_mean' and hasattr(layer, 'variables'):
        return layer.variables[2]
      if var_name == 'moving_variance' and hasattr(layer, 'variables'):
        return layer.variables[3]
    if len(layers) > 1:
      tf_logging.vlog(SHOW_PERMUTATION_INFO,
                      "Failed to distinguish variables for op_name=%s, "
                      "candidates=(%s). Usually this happens when layers have "
                      "the same name in nested models. Please consider rename "
                      "them." % (node_name,
                                 ", ".join([x.name for x in layers])))
  return None


def build_kernel_map(graph_def, layer_dict):
  """Creates a dict of op names with their variables.
  Returns:
    A dict {op_name: {var_name: var}}. The 'var' might be None, meaning the
      naming convention used in the model doesn't match our assumption.
      Depending on different ops, the valid 'var_name's are:
      * MatMul: kernel
      * Conv2D: kernel
      * BiasAdd: bias
      * FusedBatchNormV3: gamma, beta, moving_mean, moving_variance
  """
  kernel_dict = {}
  for node in graph_def['node']:
    node_name = node['name']
    node_op = node['op']

    if node_op in PERMUTABLE_2x4_ALLOWLIST:
      kernel_dict[node_name] = {}
      kernel = find_variable(node_name, node_op, 'kernel', layer_dict)
      kernel_dict[node_name]['kernel'] = kernel

    elif node_op in PERMUTABLE_2x4_INFERLIST:
      kernel_dict[node_name] = {}
      if node_op == 'BiasAdd':
        bias = find_variable(node_name, node_op, 'bias', layer_dict)
        kernel_dict[node_name]['bias'] = bias

      if node_op == 'FusedBatchNormV3':
        gamma = find_variable(node_name, node_op, 'gamma', layer_dict)
        beta = find_variable(node_name, node_op, 'beta', layer_dict)
        moving_mean = find_variable(node_name, node_op, 'moving_mean',
                                    layer_dict)
        moving_variance = find_variable(node_name, node_op, 'moving_variance',
                                        layer_dict)
        kernel_dict[node_name]['gamma'] = gamma
        kernel_dict[node_name]['beta'] = beta
        kernel_dict[node_name]['moving_mean'] = moving_mean
        kernel_dict[node_name]['moving_variance'] = moving_variance

  return kernel_dict


def build_node_map(graph_def, kernel_map, prunable_kernels):
  """Builds a dict of op nodes with their attributes.

  Returns:
    A dict of {op_name: {attr_name: attr}}. The valid 'attr_name' are:
      * op: a string of op_type
      * inputs: a list of parent node names
      * category: a string of ['allow', 'infer', 'clear', 'skip', 'deny']
      When the op_type is 'MatMul' or 'Conv2D', there is one more attr:
      * kernel: a variable tensor
      When the op_type is 'BiasAdd', there is one more attr:
      * bias: a variable tensor
      When the op_type is 'FusedBatchNormV3', there are four more attrs:
      * gamma: a variable tensor
      * beta: a variable tensor
      * moving_mean: a variable tensor
      * moving_variance: a variable tensor
      Note, all node's 'category' is 'deny' at the beginning and only when its
      variable tensor is successfully assigned, the 'category' will be switched
      to 'allow' or 'infer'.
  """
  node_map = {}
  for node in graph_def['node']:
    node_name = node['name']
    node_map[node_name] = {}
    node_map[node_name]['op'] = node['op']
    node_map[node_name]['inputs'] = node['input'] if 'input' in node else []

    node_map[node_name]['category'] = 'deny'
    if node['op'] in PERMUTABLE_2x4_ALLOWLIST:
      kernel = kernel_map[node_name]['kernel']
      if kernel is not None and kernel.ref() in prunable_kernels:
        node_map[node_name]['kernel'] = kernel
        node_map[node_name]['category'] = 'allow'

    if node['op'] in PERMUTABLE_2x4_INFERLIST:
      if node['op'] == 'BiasAdd':
        bias = kernel_map[node_name]['bias']
        if bias is not None:
          node_map[node_name]['bias'] = bias
          node_map[node_name]['category'] = 'infer'
      if node['op'] == 'FusedBatchNormV3':
        gamma = kernel_map[node_name]['gamma']
        beta = kernel_map[node_name]['beta']
        moving_mean = kernel_map[node_name]['moving_mean']
        moving_variance = kernel_map[node_name]['moving_variance']
        if not [x for x in (gamma, beta, moving_mean, moving_variance)
                if x is None]:
          node_map[node_name]['gamma'] = gamma
          node_map[node_name]['beta'] = beta
          node_map[node_name]['moving_mean'] = moving_mean
          node_map[node_name]['moving_variance'] = moving_variance
          node_map[node_name]['category'] = 'infer'

    if node['op'] in PERMUTABLE_2x4_CLEARLIST:
      node_map[node_name]['category'] = 'clear'

    if node['op'] in PERMUTABLE_2x4_SKIPLIST:
      node_map[node_name]['category'] = 'skip'

  return node_map


def check_skippable(node_name, node_map):
  """Checks if the bransh starting from node_name is skippable.

    We define the branch that will read variables as a skippable branch. The
    node_name must be an immediate parents of an allowlist or inferlist node.
    For example, these three patterns will be marked as skippable branch.
      * Conv2D<-(ReadVariableOp)
      * Conv2D<-(Cast)<-ReadVariableOp
      * Conv2D->(Const)
    These patterns will be the non-skippable:
      * Conv2D<-(Placeholder)
      * Conv2D<-(Cast)<-ReLU.
    Note, the node in the parenthesis is the node_name.
  """
  node_op = node_map[node_name]['op']
  if node_op in ('Const', 'ReadVariableOp'):
    return True

  if node_op == 'Cast':
    if 'inputs' in node_map[node_name]:
      parent_node_names = node_map[node_name]['inputs']
      if (len(parent_node_names) == 1 and
          node_map[parent_node_names[0]]['op'] == 'ReadVariableOp'):
        return True

  return False


def find_allowlist_parents_helper(node_name, node_map):
  """Helper function for find_allowlist_parents(). """
  node_category = node_map[node_name]['category']

  if node_category == 'allow':
    return [node_name]

  if node_category in ('infer', 'clear'):
    parent_node_names = node_map[node_name]['inputs']
    parents = []
    for parent_node_name in parent_node_names:
      if (node_category == 'infer' and check_skippable(parent_node_name,
                                                       node_map)):
        continue

      new_parents = find_allowlist_parents_helper(parent_node_name, node_map)
      if new_parents is None:
        return None

      parents.extend(x for x in new_parents if x not in parents)

    if len(parents) == 0:
      return None

    return parents

  return None


def find_allowlist_parents(node_name, node_map):
  """Finds all valid allowlist parent nodes of node_name.

  We define the valid allowlist parents as the allowlist nodes that are on the
  upstream paths of node_name and for each path, there is no other allowlist
  nodes in between. Note, we return an empty list if any upstream path of
  node_name is not ended with an allowlist op.

  Args:
    node_name: A node name, which must be either an allowlist or inferlist node.
    node_map: A node map. See build_node_map().

  Returns:
    A list of valid allowlist parent node names.
  """
  assert node_map[node_name]['category'] in ('allow', 'infer')

  parents = []
  parent_node_names = node_map[node_name]['inputs']
  for parent_node_name in parent_node_names:
    node_category = node_map[parent_node_name]['category']

    if check_skippable(parent_node_name, node_map):
      continue

    new_parents = find_allowlist_parents_helper(parent_node_name, node_map)
    # If any node has no valid parents, we should early exit, since we've found
    # a not-ended-with-allowlist-node branch.
    if new_parents is None:
      return []

    parents.extend(x for x in new_parents if x not in parents)

  return parents


def build_permute_map(node_map):
  """Builds a map to track the permutation on allowlist and inferlist ops.

  Args:
    node_map: A node map. See build_node_map().

  Returns:
    A dict in the form of:
      { 'node_name':
          { 'parents':[],
            'children':[],
            'c-permuted': False,
            'k-permuted': False,
            'sibling_group_index': -1
          }
      }. The 'node_name' represents ops from the allowlist or inferlist.
  """
  processed = {}

  for node_name in node_map:
    if node_map[node_name]['category'] in ('allow', 'infer'):
      if node_name not in processed:
        processed[node_name] = {}

      parent_node_names = find_allowlist_parents(node_name, node_map)

      for parent_node_name in parent_node_names:
        if parent_node_name not in processed:
          processed[parent_node_name] = {}

        if 'parents' not in processed[node_name]:
          processed[node_name]['parents'] = []
        processed[node_name]['parents'].append(parent_node_name)
        processed[node_name]['k-permuted'] = False
        processed[node_name]['c-permuted'] = False
        processed[node_name]['sibling_group_index'] = -1

        # We don't update inferlist nodes' parent nodes.
        if node_map[node_name]['category'] == 'infer':
          continue

        if 'children' not in processed[parent_node_name]:
          processed[parent_node_name]['children'] = []
        processed[parent_node_name]['children'].append(node_name)
        processed[parent_node_name]['k-permuted'] = False
        processed[parent_node_name]['c-permuted'] = False
        processed[parent_node_name]['sibling_group_index'] = -1

  return processed


def find_siblings(node_name, permute_map, found_siblings):
  """Finds all siblings of node_name.

  Returns:
    An updated sibling list by considering node_name. The siblings include the
    node_name itself.
  """
  # We don't permute the top layer of allowlist ops.
  if 'parents' not in permute_map[node_name]:
    return found_siblings

  siblings = [node_name]

  # Finds siblings that have the same parent with node_name.
  for parent_node_name in permute_map[node_name]['parents']:
    for child_node_name in permute_map[parent_node_name]['children']:
      if child_node_name != node_name:
        siblings.append(child_node_name)
  new_siblings = [x for x in siblings if x not in found_siblings]

  found_siblings.extend(new_siblings)

  # Finds siblings of the above new_siblings. They are also siblings of
  # node_name.
  for new_sibling in new_siblings:
    found_siblings = find_siblings(new_sibling, permute_map, found_siblings)

  return found_siblings


def get_weights(node_name, node_map):
  """Returns a transposed/reshaped 2D weights of node_name.

  For Conv2D, the weight is in (H,W,I,O) shape and we transpose it
  to (I, O*H*W). For MatMul, we directly return its weight in (I, O) shape.

  Returns:
    A 2D tensor from the weight of node_name, or None if the weight cannot be
    found in prunable_weights.
  """
  kernel = node_map[node_name]['kernel']
  # Since node_map[node_name]['category'] is expected to be 'allow', the
  # 'kernel' must exist.
  assert kernel is not None

  if node_map[node_name]['op'] == 'Conv2D':
    kernel = tf.transpose(kernel, perm=[2, 3, 0, 1])

  transposed_kernel = tf.reshape(kernel, shape=(kernel.shape[0], -1))
  return transposed_kernel


def search_for_good_permutation(matrix, search_device, search_time_limit):
  """Finds best permutation seq over the input dim of matrix. """
  if search_device not in ('GPU', 'CPU', 'DEBUG'):
    raise ValueError(
              "search_device=%s is not supported." % search_device)

  if search_device == 'DEBUG':
    perm = tf.range(0, matrix.shape[0])
    perm = tf.reverse(perm, axis=[0])
    return perm, ""

  # The Exhaustive_Search() and sum_after_2_to_4() expect the matrix in the
  # shape of (O, I).
  transposed_matrix = tf.transpose(matrix)
  input_dim = transposed_matrix.shape[1]

  # TODO(kaixih): Move this logic to GPU if perf issue is hit.
  original_magnitude = tf.math.reduce_sum(tf.math.abs(transposed_matrix))
  pruned_magnitude = sum_after_2_to_4(transposed_matrix.numpy())
  epsilon = 1e-3
  # We want to skip the permutation step if the pruned_magnitude is already good
  # enough.
  if (original_magnitude - pruned_magnitude) > epsilon:
    if input_dim <= 2048:
      permuted_matrix, duration, perm = Exhaustive_Search(
          transposed_matrix.numpy(), stripe_group_size=8, escape_attempts=100,
          search_device=search_device)
    else:
      permuted_matrix = transposed_matrix.numpy()
      real_swap_num = 0
      start_time = time.perf_counter()
      perm = list(range(input_dim))
      while time.perf_counter() - start_time < search_time_limit:
        src = np.random.randint(input_dim)
        dst = np.random.randint(input_dim)
        src_group = int(src / 4)
        dst_group = int(dst / 4)
        if src_group == dst_group: # channel swapping within a stripe does nothing
          continue
        new_sum, improvement = try_swap(permuted_matrix, dst, src)
        if improvement > 1e-9:
          permuted_matrix[...,[src,dst]] = permuted_matrix[...,[dst,src]]
          real_swap_num += 1
          perm[src], perm[dst] = perm[dst], perm[src]

      duration = time.perf_counter() - start_time
      tf_logging.vlog(SHOW_PERMUTATION_MORE_INFO,
                      "[TF-ASP] Finally swap {} channel pairs until the search "
                      "time limit expires.".format(real_swap_num))

    permuted_magnitude = sum_after_2_to_4(permuted_matrix)
    if (pruned_magnitude - permuted_magnitude) > epsilon:
      return None, "pruned_magnitude (%f) >= permuted_magnitude (%f)" % (
                       pruned_magnitude, permuted_magnitude)

    return perm, "permuted_magnitude (%f) >= pruned_magnitude (%f)" % (
                     permuted_magnitude, pruned_magnitude)
  else:
    return None, "pruned_magnitude (%f) >= original_magnitude (%f)" % (
                     pruned_magnitude, original_magnitude)


def find_permutation(node_name, permute_map, node_map, search_device,
                     search_time_limit, index_generator):
  """Finds the permutation sequence and update the permute_map. """
  if 'permutation' in permute_map[node_name]:
    return

  if node_map[node_name]['category'] != 'allow':
    return

  siblings = find_siblings(node_name, permute_map, [])

  sibling_weights = []
  for sibling in siblings:
    weights = get_weights(sibling, node_map)
    if weights is not None:
      sibling_weights.append(weights)

  if len(sibling_weights) != 0:
    # The weights from siblings are concatenated along the output dim. So,
    # concat_weights is in the shape of (I, n*O).
    concat_weights = tf.concat(sibling_weights, axis=1)

    permutation_seq, magnitude_info = search_for_good_permutation(
        concat_weights, search_device, search_time_limit)

    sibling_group_index = next(index_generator)
    # Broadcast the permutation sequence to all siblings.
    for sibling in siblings:
      permute_map[sibling]['permutation'] = permutation_seq
      permute_map[sibling]['sibling_group_index'] = sibling_group_index

    tf_logging.vlog(SHOW_PERMUTATION_MORE_INFO,
        "[TF-ASP] Permute-Siblings: %s (%s: %s)" % (
            ",".join([x for x in siblings]),
            "Skipped" if permutation_seq is None else "Enabled",
            magnitude_info))


def permute_C(node_name, node_map, permutation_seq):
  """Permutes the input dim of the weights from node_name. """
  node_op = node_map[node_name]['op']
  # Since permutation_seq exists, node_map musts be from the allowlist.
  assert node_op in PERMUTABLE_2x4_ALLOWLIST

  kernel = node_map[node_name]['kernel']
  assert kernel is not None

  if node_op == 'Conv2D':
    transposed_kernel = tf.transpose(kernel, perm=[2, 3, 0, 1])
    transposed_shape = transposed_kernel.shape
    transposed_kernel = tf.reshape(transposed_kernel,
                                   shape=(transposed_shape[0], -1))

    shuffled_kernel = tf.gather(transposed_kernel, permutation_seq)

    transposed_kernel = tf.reshape(shuffled_kernel, shape=transposed_shape)
    recovered_kernel = tf.transpose(transposed_kernel, perm=[2, 3, 0, 1])

    kernel.assign(recovered_kernel)

  if node_op == 'MatMul':
    shuffled_kernel = tf.gather(kernel, permutation_seq)

    kernel.assign(shuffled_kernel)

  tf_logging.vlog(SHOW_PERMUTATION_MORE_INFO,
      "[TF-ASP] Permute-C: node_name=%s" % node_name)


def permute_K_impl(node_name, node_map, permutation_seq, trigger_node):
  """Permutes the output dim of the weights from node_name. """
  node_op = node_map[node_name]['op']

  if node_op in PERMUTABLE_2x4_ALLOWLIST:
    kernel = node_map[node_name]['kernel']
    assert kernel is not None
    new_kernel = tf.gather(kernel, permutation_seq, axis=-1)
    kernel.assign(new_kernel)

  if node_op in PERMUTABLE_2x4_INFERLIST:
    if node_op == "BiasAdd":
      bias = node_map[node_name]['bias']
      assert bias is not None
      new_bias = tf.gather(bias, permutation_seq)
      bias.assign(new_bias)

    if node_op == 'FusedBatchNormV3':
      gamma = node_map[node_name]['gamma']
      beta = node_map[node_name]['beta']
      moving_mean = node_map[node_name]['moving_mean']
      moving_variance = node_map[node_name]['moving_variance']
      assert not [x for x in (gamma, beta, moving_mean, moving_variance) if x is
                  None]
      new_gamma = tf.gather(gamma, permutation_seq)
      new_beta = tf.gather(beta, permutation_seq)
      new_moving_mean = tf.gather(moving_mean, permutation_seq)
      new_moving_variance = tf.gather(moving_variance, permutation_seq)

      gamma.assign(new_gamma)
      beta.assign(new_beta)
      moving_mean.assign(new_moving_mean)
      moving_variance.assign(new_moving_variance)

  tf_logging.vlog(SHOW_PERMUTATION_MORE_INFO,
      "[TF-ASP] Permute-K: node_name=%s, permute_seq from %s" % (
          node_name, trigger_node))


def permute_K_helper(node_name, permute_map, node_map, permutation_seq,
                     trigger_node):
  """Permutes output dims of weights from node_name's upstream ops. """
  parent_node_names = node_map[node_name]['inputs']
  for parent_node_name in parent_node_names:
    node_category = node_map[parent_node_name]['category']
    # Finds an allowlist op.
    if (node_category == 'allow' and
        not permute_map[parent_node_name]['k-permuted']):
      permute_K_impl(parent_node_name, node_map, permutation_seq, trigger_node)
      permute_map[parent_node_name]['k-permuted'] = True
    # Finds an inferlist op that hasn't been permuted yet.
    elif (node_category == 'infer' and
          not permute_map[parent_node_name]['k-permuted']):
      permute_K_impl(parent_node_name, node_map, permutation_seq, trigger_node)
      permute_map[parent_node_name]['k-permuted'] = True
      permute_K_helper(parent_node_name, permute_map, node_map, permutation_seq,
                       trigger_node)
    # Finds a clearlist op and passes the permutation seq through.
    elif node_category == 'clear':
      permute_K_helper(parent_node_name, permute_map, node_map, permutation_seq,
                       trigger_node)


def apply_permutation(node_name, permute_map, node_map):
  """Applies the permutation to node_name.

  This function permutes the input dim of node_name (c-permute) and permutes the
  output dim of its upstream nodes (k-permute).
  """
  if 'permutation' in permute_map[node_name]:
    permutation_seq = permute_map[node_name]['permutation']
    if permutation_seq is None:
      return
    if not permute_map[node_name]['c-permuted']:
      permute_C(node_name, node_map, permutation_seq)
      permute_map[node_name]['c-permuted'] = True

    # When "permutation" exists, the sequence must be propagated upstream,
    # meaning the parent nodes must exist and be permutable.
    permute_K_helper(node_name, permute_map, node_map, permutation_seq,
                     node_name)


def count_c_permuted(permute_map):
  """Counts how many layers are C-permuted. """
  count = 0
  for node_name in permute_map:
    if permute_map[node_name] and permute_map[node_name]['c-permuted']:
      count += 1
  return count


def permute_model(model, input_shapes, prunable_kernels, search_device,
                  search_time_limit, plot_to_file):
  """Permute weights from all eligible ops in the model.

  This function will first traverse the GraphDef obtained from the model. Note,
  the GraphDef contains a graph of operations rather than layers. So, it will
  locate weights from all eligible operations for pruning and then conduct
  the permutation. The permutation includes C-permute which permutes the input
  dim of the current weights and K-permute which permutes the output dim of the
  previous weights so as to match the results of C-permute.

  Args:
    model: A built model.
    input_shapes: A tuple or a list of tuples representing the input tensor
      shapes.
    prunable_kernels: A set of kernel refs that are pruning.
    search_device: A string representing where the permutation searching occurs.
      Valid strings are ['GPU'(default), 'CPU', 'DEBUG'].
    plot_to_file: A string of file name to plot the colored op graph.
  """
  graph_def = get_graph_def(model, input_shapes)
  layer_dict = build_layer_dict(model)
  kernel_map = build_kernel_map(graph_def, layer_dict)

  tf_logging.vlog(SHOW_PERMUTATION_DEBUG_INFO,
      "[TF-ASP] DEBUG kernel_map:\n" + pprint.pformat(kernel_map, indent=2))

  node_map = build_node_map(graph_def, kernel_map, prunable_kernels)

  tf_logging.vlog(SHOW_PERMUTATION_DEBUG_INFO,
      "[TF-ASP] DEBUG node_map:\n" + pprint.pformat(node_map, indent=2))

  permute_map = build_permute_map(node_map)

  # After find_permutation(), the item in permute_map may not have "parents" and
  # "permutation" if the node_name is the top layer, meaning it is only
  # eligible for k-permute; the item may not have "children" if the node_name is
  # the bottom layer.
  index_generator = count()
  for node_name in permute_map:
    find_permutation(node_name, permute_map, node_map, search_device,
                     search_time_limit, index_generator)

  tf_logging.vlog(SHOW_PERMUTATION_DEBUG_INFO,
      "[TF-ASP] DEBUG permute_map (prolog):\n" + pprint.pformat(
          permute_map, indent=2, compact=True))

  for node_name in permute_map:
    apply_permutation(node_name, permute_map, node_map)

  tf_logging.vlog(SHOW_PERMUTATION_DEBUG_INFO,
      "[TF-ASP] DEBUG permute_map (epilog):\n" + pprint.pformat(
          permute_map, indent=2, compact=True))

  if plot_to_file:
    plot_ops_graph(node_map, permute_map, plot_to_file)

  tf_logging.vlog(SHOW_PERMUTATION_INFO,
      "[TF-ASP] %d/%d variables (Conv2D or MatMul) are permuted!\n" % (
          count_c_permuted(permute_map), len(prunable_kernels)))


def check_pydot():
  """Returns True if PyDot and Graphviz are available."""
  if pydot is None:
    return False
  try:
    # Attempt to create an image of a blank graph
    # to check the pydot/graphviz installation.
    pydot.Dot.create(pydot.Dot())
    return True
  except (OSError, pydot.InvocationException):
    return False


def add_edge(dot, src, dst):
  """Adds edge from src to dst. """
  if not dot.get_edge(src, dst):
    dot.add_edge(pydot.Edge(src, dst))


def plot_ops_graph(node_map, permute_map, to_file):
  """Converts ops to dot format and save to a file. """
  if not check_pydot():
    message = (
        'You must install pydot (`pip install pydot`) '
        'and install graphviz '
        '(see instructions at https://graphviz.gitlab.io/download/) ',
        'for plot_to_file option to work.')
    raise ImportError(message)

  dot = pydot.Dot()
  dot.set('rankdir', 'TB')
  dot.set('concentrate', True)
  dot.set('dpi', 96)
  dot.set_node_defaults(shape='record')

  # A handy knob to indicate whether the plot contains the skiplist ops.
  contains_skiplist_nodes = True

  # Add all the nodes to the dot.
  for node_name in node_map:
    node_category = node_map[node_name]['category']

    fillcolor = 'red'
    if node_category == 'allow':
      fillcolor = 'green'
    elif node_category == 'infer':
      fillcolor = 'orange'
    elif node_category == 'clear':
      fillcolor = 'yellow'
    elif node_category == 'skip':
      fillcolor = 'grey'

    if not contains_skiplist_nodes and node_category == 'skip':
      continue

    def format_shape(shape):
      return str(shape).replace(str(None), 'None')

    label = node_map[node_name]['op']
    if node_category == 'allow':
      kernel = node_map[node_name]['kernel']
      assert kernel is not None
      kernel_shape = format_shape(kernel.shape)
      prunable_flag = 'S'

      sibling_group_flag = -1
      permute_k_flag = ''
      permute_c_flag = ''
      if node_name in permute_map:
        if permute_map[node_name]:
          sibling_group_flag = permute_map[node_name]['sibling_group_index']
          permute_k_flag = 'K' if permute_map[node_name]['k-permuted'] else ''
          permute_c_flag = 'C' if permute_map[node_name]['c-permuted'] else ''
        else:
          sibling_group_flag, permute_k_flag, permute_c_flag  = (-1, '', '')

      label = '{%s (%d, %s%s, %s)| kernel=%s}' % (
          label, sibling_group_flag, permute_k_flag, permute_c_flag,
          prunable_flag, kernel_shape)

    if node_category == 'infer':
      if node_name in permute_map and permute_map[node_name]:
        label += ' (K)' if permute_map[node_name]['k-permuted'] else ''

    if node_category == 'skip':
      node = pydot.Node(node_name, label=label[0:2], style='filled',
                        fillcolor=fillcolor, shape='circle', fontsize=10)
    else:
      node = pydot.Node(node_name, label=label, style='filled',
                        fillcolor=fillcolor)
    dot.add_node(node)

  max_edges = 9999
  edge_count = 0
  try:
    # Create edges for these nodes.
    for dst_node_name in node_map:
      for src_node_name in node_map[dst_node_name]['inputs']:
        # We skip the src nodes if their name start with '^'. It seems they are
        # some virtual nodes and we are not interested in them.
        if (src_node_name.startswith('^') or (src_node_name not in node_map) or
            (not contains_skiplist_nodes and
             (node_map[dst_node_name]['category'] == 'skip' or
              node_map[src_node_name]['category'] == 'skip'))):
          continue
        add_edge(dot, src_node_name, dst_node_name)
        edge_count += 1
        if edge_count >= max_edges:
          raise StopIteration
  except StopIteration:
    print("[TF-ASP] The op graph is too large to plot. Only up to %d edges are"
          " plotted." % max_edges)

  file_name, extension = os.path.splitext(to_file)
  if not extension:
    extension = 'png'
  else:
    extension = extension[1:]
  dot.write(file_name + '.' + extension, format=extension)
  print("[TF-ASP] The op graph is plotted to %s." % to_file)

