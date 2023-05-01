# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import inspect
import numpy as np
import os
import tempfile
import tensorflow as tf
from atex.structured_sparsity import tf_asp
import shutil

from tensorflow.keras import layers, optimizers
from tensorflow.python.platform import test


def GetSingleLayerConfigs():
  """Gets all valid tests for single layer model.

  Returns:
    All the test configs as tuples of (layer_name, input_shape, output_dim).
  """
  layers = ["Dense", "Dense", "Dense", "Dense","Dense", "Dense",
            "Conv2D", "Conv2D"]
  input_shapes = [(32,), (64,), (128,), (127,), (126,), (125,),
                  (64, 64, 32), (32, 32, 64)]
  output_dims = [8, 16, 32, 8, 8, 8,
                 16, 32]
  for l, i, o in zip(layers, input_shapes, output_dims):
    yield l, i, o


def GetConvNetModel():
  """Gets an example convnet model. """
  input1 = layers.Input(shape=(28, 28, 16))
  conv1_1 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                          name="conv1_1")
  conv1_2 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                          name="conv1_2")
  conv2_1 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                          name="conv2_1")
  conv3 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                        name="conv3")
  add1 = layers.Add(name="add1")
  batch_norm1 = layers.BatchNormalization(
      beta_initializer='uniform', gamma_initializer='uniform',
      moving_mean_initializer='uniform',
      moving_variance_initializer='uniform', name="batch_norm1")
  batch_norm2 = layers.BatchNormalization(
      beta_initializer='uniform', gamma_initializer='uniform',
      moving_mean_initializer='uniform',
      moving_variance_initializer='uniform', name="batch_norm2")
  batch_norm3 = layers.BatchNormalization(
      beta_initializer='uniform', gamma_initializer='uniform',
      moving_mean_initializer='uniform',
      moving_variance_initializer='uniform', name="batch_norm3")
  relu1 = layers.ReLU(name="relu1")

  y1_1 = conv1_1(input1)
  y1_1 = batch_norm1(y1_1)
  y1_1 = relu1(y1_1)
  y1_2 = conv1_2(y1_1)
  y1_2 = batch_norm2(y1_2)

  y2_1 = conv2_1(input1)
  y2_1 = batch_norm3(y2_1)

  y2 = add1([y2_1, y1_2])
  output1 = conv3(y2)
  model = tf.keras.models.Model(inputs=input1, outputs=output1)
  return model


def GetInferlistModel(layer_names, input_shape, output_dim):
  """Gets a sequential model with given layers.
  Args:
    layer_names: A list of supported layer names. The first and last name should
      be conv* or dense* at the same time. The middle names should be
      batch_norm*, relu*.
    input_shape: A tuple of integers representing the input shape.
    output_dim: An integer representing the output dim of layers.
  Returns:
    A sequential model.
  Raises:
    A value error if unsupported names are encountered.
  """
  model = tf.keras.Sequential(name="sequential")
  if len(layer_names) == 0:
    raise ValueError("The layer_names could not be empty.")

  model.add(layers.Input(shape=input_shape))

  for i, layer_name in enumerate(layer_names):
    if layer_name.startswith("conv"):
      if i + 1 < len(layer_names) and layer_names[i + 1].startswith("bias_add"):
        use_bias=True
      else:
        use_bias=False
      model.add(layers.Conv2D(output_dim, 3, padding='same', use_bias=use_bias,
                              name=layer_name))

    elif layer_name.startswith("dense"):
      if i + 1 < len(layer_names) and layer_names[i + 1].startswith("bias_add"):
        use_bias=True
      else:
        use_bias=False
      model.add(layers.Dense(output_dim, use_bias=use_bias, name=layer_name))

    elif layer_name.startswith("bias_add"):
      continue

    elif layer_name.startswith("batch_norm"):
      model.add(layers.BatchNormalization(
                    beta_initializer='uniform', gamma_initializer='uniform',
                    moving_mean_initializer='uniform',
                    moving_variance_initializer='uniform', name=layer_name))

    elif layer_name.startswith("relu"):
      model.add(layers.ReLU(name=layer_name))
    else:
      raise ValueError(
          "The layer_names contains unsupported layer_name: %s" % layer_name)
  return model

def GetInferlistModelConfigs():
  """Gets all valid tests for inferlist model.
  """
  layers = [
      ("conv1", "batch_norm1", "conv2"),
      ("conv1", "bias_add", "conv2"),
      ("dense1", "bias_add", "dense2"),
      ("conv1", "bias_add", "batch_norm1", "batch_norm2", "relu1", "conv2"),
      ("dense1", "bias_add", "relu1", "dense2"),
      ("dense1", "bias_add", "dense2"),
      ("dense1", "bias_add", "dense2"),
      ]

  expected_logs = [
      ("Permute-C: node_name=sequential/conv2/Conv2D",
       "Permute-K: node_name=sequential/batch_norm1/FusedBatchNormV3",
       "Permute-K: node_name=sequential/conv1/Conv2D"),
      ("Permute-C: node_name=sequential/conv2/Conv2D",
       "Permute-K: node_name=sequential/conv1/BiasAdd",
       "Permute-K: node_name=sequential/conv1/Conv2D"),
      ("Permute-C: node_name=sequential/dense2/MatMul",
       "Permute-K: node_name=sequential/dense1/BiasAdd",
       "Permute-K: node_name=sequential/dense1/MatMul"),
      ("Permute-C: node_name=sequential/conv2/Conv2D",
       "Permute-K: node_name=sequential/batch_norm2/FusedBatchNormV3",
       "Permute-K: node_name=sequential/batch_norm1/FusedBatchNormV3",
       "Permute-K: node_name=sequential/conv1/BiasAdd",
       "Permute-K: node_name=sequential/conv1/Conv2D"),
      ("Permute-C: node_name=sequential/dense2/MatMul",
       "Permute-K: node_name=sequential/dense1/BiasAdd",
       "Permute-K: node_name=sequential/dense1/MatMul"),
      ("Permute-C: node_name=sequential/dense2/MatMul",
       "Permute-K: node_name=sequential/dense1/BiasAdd",
       "Permute-K: node_name=sequential/dense1/MatMul"),
      ("Permute-C: node_name=sequential/dense2/MatMul",
       "Permute-K: node_name=sequential/dense1/BiasAdd",
       "Permute-K: node_name=sequential/dense1/MatMul"),
      ]
  input_shapes = [
      (28, 28, 16),
      (28, 28, 16),
      (64,),
      (28, 28, 16),
      (64,),
      (1024,),
      (32,),
      ]
  output_dims = [32, 32, 32, 32, 32, 512, 16]
  devices = ["GPU", "GPU", "GPU", "GPU", "GPU", "GPU", "CPU"]
  for c, l, i, o, d in zip(layers, expected_logs, input_shapes,
                           output_dims, devices):
    yield c, l, i, o, d


class TfAspOptimizerTest(test.TestCase):

  def _CheckMask(self, mask):
    """Checks if every 4 values contain 2 zeros. """
    mask_ndims = len(mask.shape)
    # For Dense: mask's shape (I, O). For Conv2D: mask's shape (H, W, I, O). We
    # need to transpose them to (None, I) for better access since the pruning is
    # along the I dim.
    if mask_ndims == 2:
      mask = tf.transpose(mask)
    elif mask_ndims == 4:
      mask = tf.transpose(mask, perm=[0, 1, 3, 2])
      mask = tf.reshape(mask, shape=(-1, mask.shape[-1]))

    result = True
    ngroups = mask.shape[1] // 4
    for row in range(mask.shape[0]):
      for col in range(0, ngroups * 4, 4):
        one_mask = mask[row, col:col+4]
        result = result and (tf.math.reduce_sum(one_mask) == 2)
      if ngroups * 4 < mask.shape[1]:
        one_mask = mask[row, ngroups*4:]
        result = result and (tf.math.reduce_sum(one_mask) <= 2)

    self.assertEqual(result, True)


  def testPrunedSparsitySingleLayer(self):
    for layer, input_shape, output_dim in GetSingleLayerConfigs():
      input_1 = layers.Input(shape=input_shape)
      if layer == "Dense":
        layer_1 = layers.Dense(output_dim, name="dense_1")
      elif layer == "Conv2D":
        layer_1 = layers.Conv2D(output_dim, (3, 3), name="conv_1")
      output_1 = layer_1(input_1)
      model = tf.keras.models.Model(inputs=input_1, outputs=output_1)

      opt = optimizers.legacy.SGD(learning_rate=0.2, momentum=1.0)
      opt = tf_asp.AspOptimizerWrapperV2(
                opt, model, permute=False, padding=True,
                plot_to_file=inspect.currentframe().f_code.co_name)

      if layer == "Dense":
        batched_shape = (5,) + input_shape
      elif layer == "Conv2D":
        batched_shape = (5,) + input_shape
      x = tf.random.normal(shape=batched_shape)

      # Run the train step once to trigger the mask compute.
      with tf.GradientTape(persistent=True) as tape:
        y = model(x)
        loss = tf.reduce_mean(y)
      grads = tape.gradient(loss, model.variables)
      opt.apply_gradients(zip(grads, model.variables))

      mask = opt.get_slot(layer_1.kernel, "mask")

      self._CheckMask(mask)


  def testMasksCanBeUpdatedOnlyOnceInTrainLoop(self):
    model = GetConvNetModel()
    opt = optimizers.legacy.SGD(learning_rate=0.2, momentum=1.0)
    opt = tf_asp.AspOptimizerWrapperV2(
              opt, model, permute=False, padding=True,
              plot_to_file=inspect.currentframe().f_code.co_name)

    @tf.function
    def train_step(x):
      with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.reduce_sum(y)
      grads = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(grads, model.trainable_variables))
      return loss

    x_train = tf.random.normal(shape=(100, 28, 28, 16))

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(10)

    masks_ref = []
    masks = []

    for step, x in enumerate(train_dataset):
      loss = train_step(x)
      # Inital train step creates and updates the masks. Following train steps
      # shouldn't change the masks.
      if step == 0:
        for layer in model.layers:
          if isinstance(layer, layers.Conv2D):
            masks_ref.append(tf.identity(opt.get_slot(layer.kernel, "mask")))
      if step == 10:
        for layer in model.layers:
          if isinstance(layer, layers.Conv2D):
            masks.append(opt.get_slot(layer.kernel, "mask"))
        break

    for mask_ref, mask in zip(masks_ref, masks):
      self.assertAllEqual(mask_ref, mask)


  def testMasksCanBeUpdatedOnlyOnceInModelFit(self):
    model = GetConvNetModel()
    opt = optimizers.legacy.SGD(learning_rate=0.2, momentum=1.0)
    opt = tf_asp.AspOptimizerWrapperV2(
              opt, model, permute=False, padding=True,
              plot_to_file=inspect.currentframe().f_code.co_name)

    model.compile(optimizer=opt, loss="mse")

    x_train = tf.random.normal(shape=(100, 28, 28, 16))
    y_train = tf.random.normal(shape=(100, 28, 28, 32))

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(10)

    # Inital train step creates and updates the masks.
    model.fit(train_dataset, epochs=1, steps_per_epoch=1)
    masks_ref = []
    for layer in model.layers:
      if isinstance(layer, layers.Conv2D):
        masks_ref.append(tf.identity(opt.get_slot(layer.kernel, "mask")))

    # Following train steps shouldn't change the masks.
    model.fit(train_dataset, initial_epoch=1, epochs=10, steps_per_epoch=1)

    masks = []
    for layer in model.layers:
      if isinstance(layer, layers.Conv2D):
        masks.append(opt.get_slot(layer.kernel, "mask"))

    for mask_ref, mask in zip(masks_ref, masks):
      self.assertAllEqual(mask_ref, mask)


  def testInnerOptimizerWithIncreasingIterations(self):
    model = GetConvNetModel()
    inner_opt = optimizers.legacy.SGD(learning_rate=0.2, momentum=1.0)
    opt = tf_asp.AspOptimizerWrapperV2(
              inner_opt, model, permute=False, padding=True,
              plot_to_file=inspect.currentframe().f_code.co_name)

    @tf.function
    def train_step(x):
      with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.reduce_sum(y)
      grads = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(grads, model.trainable_variables))
      return loss

    x_train = tf.random.normal(shape=(100, 28, 28, 16))

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(10)

    for step, x in enumerate(train_dataset):
      loss = train_step(x)
      self.assertEqual(opt.iterations, inner_opt.iterations)


  def testInnerOptimizerHyperparameters(self):
    model = GetConvNetModel()
    inner_opt = optimizers.legacy.Adam(learning_rate=0.2)
    opt = tf_asp.AspOptimizerWrapperV2(
              inner_opt, model, permute=False, padding=True)

    # Check all hyperparameters in inner_opt._hyper
    for attr in inner_opt._hyper:
      self.assertEqual(getattr(inner_opt, attr), getattr(opt, attr))

    # Check all setattr of any optimizer can affect both.
    opt.beta_1 = 0.5
    inner_opt.beta_2 = 0.6
    self.assertEqual(inner_opt.beta_1, opt.beta_1)
    self.assertEqual(inner_opt.beta_2, opt.beta_2)

    # Check non-hyperparams.
    self.assertTrue(hasattr(inner_opt, 'epsilon'))
    self.assertFalse(hasattr(opt, 'epsilon'))


  def _CheckPermuteLogs(self, model, expected_logs, input_shapes,
                        expected_num=None,
                        search_device='GPU',
                        search_time_limit=5,
                        logger_level=tf_asp.SHOW_PERMUTATION_MORE_INFO,
                        logger_capture_level=tf_asp.SHOW_PERMUTATION_MORE_INFO,
                        plot_to_file='test.png'):
    tf.get_logger().setLevel(logger_level)

    if not isinstance(input_shapes, list):
      input_shapes = [input_shapes]

    inputs = []
    for input_shape in input_shapes:
      inputs.append(tf.random.normal(shape=(10,) + input_shape))

    if len(inputs) == 1:
      inputs = inputs[0]

    expected = model(inputs)

    opt = optimizers.legacy.SGD(learning_rate=0.2, momentum=1.0)
    # The permute is triggered during the init stage of the ASP wrapper.
    with self.assertLogs(level=logger_capture_level) as cm:
      opt = tf_asp.AspOptimizerWrapperV2(
                opt, model, permute=True, padding=True,
                search_device=search_device,
                search_time_limit=search_time_limit,
                input_shapes=(None,) + input_shape,
                plot_to_file=plot_to_file)

    matches = []
    for log in cm.output:
      for expected_log in expected_logs:
        matches.append(expected_log in log)

    if expected_num:
      self.assertEqual(sum(matches), expected_num)
    else:
      self.assertEqual(sum(matches), len(expected_logs))

    result = model(inputs)
    self.assertAllClose(expected, result, rtol=1e-2, atol=1e-2)


  def testPermuteGraphWithConvNet(self):
    model = GetConvNetModel()

    expected_logs = (
        "Permute-C: node_name=model/conv1_2/Conv2D",
        "Permute-C: node_name=model/conv3/Conv2D",
        "Permute-K: node_name=model/batch_norm1/FusedBatchNormV3",
        "Permute-K: node_name=model/batch_norm2/FusedBatchNormV3",
        "Permute-K: node_name=model/batch_norm3/FusedBatchNormV3",
        "Permute-K: node_name=model/conv1_1/Conv2D",
        "Permute-K: node_name=model/conv1_2/Conv2D",
        "Permute-K: node_name=model/conv2_1/Conv2D")

    self._CheckPermuteLogs(model, expected_logs, (28, 28, 16),
                           plot_to_file=inspect.currentframe().f_code.co_name)


  def testPermuteGraphWithComplexSiblings(self):
    input1 = layers.Input(shape=(28, 28, 16))
    conv1_1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            bias_initializer='uniform', name="conv1_1")
    conv1_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            bias_initializer='uniform', name="conv1_2")
    conv1_3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            bias_initializer='uniform', name="conv1_3")
    conv2_1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            bias_initializer='uniform', name="conv2_1")
    conv2_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            bias_initializer='uniform', name="conv2_2")
    conv3_1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            bias_initializer='uniform', name="conv3_1")
    conv3_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            bias_initializer='uniform', name="conv3_2")
    conv4_1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            bias_initializer='uniform', name="conv4_1")
    conv4_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            bias_initializer='uniform', name="conv4_2")
    add1 = layers.Add(name="add1")
    add2 = layers.Add(name="add2")

    y1_1 = conv1_1(input1)
    y1_2 = conv1_2(y1_1)
    y1_3 = conv1_3(y1_2)
    y1 = add1([y1_2, y1_3])
    y2_1 = conv2_1(input1)
    y2_2 = conv2_2(y2_1)
    y3_1 = conv3_1(y2_2)
    y3_2 = conv3_2(y3_1)
    y2 = add2([y2_2, y3_2, y1])
    y4_1 = conv4_1(y2)
    output1 = conv4_2(y4_1)
    model = tf.keras.models.Model(inputs=input1, outputs=output1)

    expected_logs = (
        "Permute-Siblings: model/conv2_2/Conv2D",
        "Permute-Siblings: model/conv1_2/Conv2D",
        "Permute-Siblings: model/conv1_3/Conv2D,model/conv4_1/Conv2D," \
            "model/conv3_1/Conv2D",
        "Permute-Siblings: model/conv3_2/Conv2D",
        "Permute-Siblings: model/conv4_2/Conv2D")

    self._CheckPermuteLogs(model, expected_logs, (28, 28, 16),
                           plot_to_file=inspect.currentframe().f_code.co_name)


  def testPermuteGraphWithInferlistOps(self):
    for i, (layers, expected_log, input_shape, output_dim, device) in \
        enumerate(GetInferlistModelConfigs()):
      model = GetInferlistModel(layers, input_shape, output_dim)
      self._CheckPermuteLogs(
          model, expected_log, input_shape, search_device=device,
          plot_to_file=inspect.currentframe().f_code.co_name + str(i))


  def testPermuteGraphWithInferlistOpAndNoEndingAllowlistOps(self):
    tf.get_logger().setLevel(tf_asp.SHOW_PERMUTATION_INFO)
    # The conv2 has two upstream branches: conv1 and batch_norm1. The
    # batch_norm1 branch is not ended with any allowlist nodes, making this
    # graph unsupported by permutation, though the conv1 branch looks good.
    shape1 = (28, 28, 128)
    shape2 = (28, 28, 16)
    x1 = layers.Input(shape=shape1)
    x2 = layers.Input(shape=shape2)
    conv1 = layers.Conv2D(16, 2, padding='same', use_bias=False, name="conv1")
    batch_norm1 = layers.BatchNormalization(name='batch_norm1')
    conv2 = layers.Conv2D(8, 2, padding='same', use_bias=False, name="conv2")
    add1 = layers.Add(name="add1")

    y1 = conv1(x1)
    y2 = batch_norm1(x2)
    y3 = add1([y1, y2])
    y4 = conv2(y3)
    model = tf.keras.models.Model(inputs=[x1, x2], outputs=y4)

    expected_log = ["0/2 variables (Conv2D or MatMul) are permuted!"]
    self._CheckPermuteLogs(model, expected_log, [shape1, shape2],
                           plot_to_file=inspect.currentframe().f_code.co_name)


  def testPermuteGraphWithClearlistOpAndNoEndingAllowlistOps(self):
    tf.get_logger().setLevel(tf_asp.SHOW_PERMUTATION_INFO)
    # The conv2 has two upstream branches: conv1 and relu1. The relu1 branch is
    # not ended with any allowlist nodes, making this graph unsupported by
    # permutation, though the conv1 branch looks good.
    shape1 = (28, 28, 128)
    shape2 = (28, 28, 16)
    x1 = layers.Input(shape=shape1)
    x2 = layers.Input(shape=shape2)
    conv1 = layers.Conv2D(16, 2, padding='same', use_bias=False, name="conv1")
    relu1 = layers.ReLU(name='relu1')
    conv2 = layers.Conv2D(8, 2, padding='same', use_bias=False, name="conv2")
    add1 = layers.Add(name="add1")

    y1 = conv1(x1)
    y2 = relu1(x2)
    y3 = add1([y1, y2])
    y4 = conv2(y3)
    model = tf.keras.models.Model(inputs=[x1, x2], outputs=y4)

    expected_log = ["0/2 variables (Conv2D or MatMul) are permuted!"]
    self._CheckPermuteLogs(model, expected_log, [shape1, shape2],
                           plot_to_file=inspect.currentframe().f_code.co_name)


  def testPermuteGraphWithUnsupportedOps(self):
    tf.get_logger().setLevel(tf_asp.SHOW_PERMUTATION_INFO)
    # The conv2 has two upstream branches: conv1_1 and conv1_2. The conv1_1
    # branch contains an unsupported "Reshape" op, making this graph
    # unsupported for permutation, though the conv1_2 branch looks good.
    shape1 = (28, 28, 128)
    shape2 = (15, 57, 128)
    x1 = layers.Input(shape=shape1)
    x2 = layers.Input(shape=shape2)
    conv1_1 = layers.Conv2D(16, 2, padding='same', use_bias=False,
                            name="conv1_1")
    conv1_2 = layers.Conv2D(16, 2, padding='valid', use_bias=False,
                            name="conv2_1")
    conv2 = layers.Conv2D(8, 2, padding='same', use_bias=False, name="conv2")
    add1 = layers.Add(name='add1')

    y1_1 = conv1_1(x1)
    old_shape = y1_1.shape
    new_shape = (-1, old_shape[1] // 2, old_shape[2] * 2, old_shape[3])
    y1_1 = tf.reshape(y1_1, shape=new_shape)
    y1_2 = conv1_2(x2)
    y1 = add1([y1_1, y1_2])
    y2 = conv2(y1)
    model = tf.keras.models.Model(inputs=[x1, x2], outputs=y2)

    expected_log = ["0/3 variables (Conv2D or MatMul) are permuted!"]
    self._CheckPermuteLogs(model, expected_log, [shape1, shape2],
                           plot_to_file=inspect.currentframe().f_code.co_name)


  def testAutomaticalSkipPermutation(self):
    # TODO(kaixih):
    self.skipTest("The second run cannot skip the permutation. Need to debug.")

    tf.get_logger().setLevel(tf_asp.SHOW_PERMUTATION_INFO)
    input_shape = (512,)
    model = GetInferlistModel(["dense1", "bias_add", "dense2"],
                              input_shape=input_shape, output_dim=128)
    opt = tf.keras.optimizers.legacy.SGD(learning_rate=0.2, momentum=1.0)

    # We wrap the optimizer for the first time to trigger the permutation.
    with self.assertLogs(level=tf_asp.SHOW_PERMUTATION_INFO) as cm:
      opt = tf_asp.AspOptimizerWrapperV2(
                opt, model, permute=True, padding=True,
                plot_to_file=inspect.currentframe().f_code.co_name)

    expected_log = "1/2 variables (Conv2D or MatMul) are permuted!"
    matches = []
    for log in cm.output:
      matches.append(expected_log in log)
    self.assertIn(True, matches)

    @tf.function
    def train_step(x):
      with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.reduce_sum(y)
      grads = tape.gradient(loss, model.trainable_variables)
      opt.apply_gradients(zip(grads, model.trainable_variables))
      return loss

    x_train = tf.random.normal(shape=(100,) + input_shape)

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(10)

    # These train steps trigger the pruning of the weights. Note, we just launch
    # one train step to ensure the updated weights are not exploded, since we
    # don't use any activation in the model.
    for step, x in enumerate(train_dataset):
      loss = train_step(x)
      if step == 0:
        break

    # We wrap the optimizer again to trigger another round of permutation.
    # However, the actual permutation should be automatically skipped, since the
    # weights have already been pruned and the permutation won't improve the
    # magnitude.
    with self.assertLogs(level=tf_asp.SHOW_PERMUTATION_MORE_INFO) as cm:
      opt = tf_asp.AspOptimizerWrapperV2(
                opt, model, permute=True, padding=True,
                plot_to_file=inspect.currentframe().f_code.co_name)

    expected_log = "0/2 variables (Conv2D or MatMul) are permuted!"
    matches = []
    for log in cm.output:
      matches.append(expected_log in log)
    self.assertIn(True, matches)


  def testPermuteWithSubclassedModel(self):
    class SubclassedModel(tf.keras.Model):
      def __init__(self, name):
        super(SubclassedModel, self).__init__(name=name)
        self.conv1_1 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                                     name="conv1_1")
        self.conv1_2 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                                     name="conv1_2")
        self.batch_norm1 = layers.BatchNormalization(
            beta_initializer='uniform', gamma_initializer='uniform',
            moving_mean_initializer='uniform',
            moving_variance_initializer='uniform', name="batch_norm1")
      def call(self, x):
        y1_1 = self.conv1_1(x)
        y1_1 = self.batch_norm1(y1_1)
        y1_2 = self.conv1_2(y1_1)
        return y1_2

    model = SubclassedModel(name='subclassed')
    input_shape = (12, 12, 16)
    model.build(input_shape=(None,) + input_shape)

    expected_logs = [
        "Permute-C: node_name=subclassed/conv1_2/Conv2D",
        "Permute-K: node_name=subclassed/batch_norm1/FusedBatchNormV3",
        "Permute-K: node_name=subclassed/conv1_1/Conv2D"
    ]
    self._CheckPermuteLogs(model, expected_logs, input_shape,
                           plot_to_file=inspect.currentframe().f_code.co_name)


  def testPermuteMixedApisWithBrokenInferlistOp(self):
    # A simple model of Conv2D->BatchNorm->Conv2D. Since the first Conv2D is not
    # from a keras layer, none of the layers would be permuted.
    class SubclassedModel(tf.keras.Model):
      def __init__(self, name):
        super(SubclassedModel, self).__init__(name=name)
        v_init = tf.random_normal_initializer()
        self.filter = tf.Variable(
            initial_value=v_init(shape=(3, 3, 16, 32), dtype='float32'),
            trainable=True)
        self.conv_layer = layers.Conv2D(
            32, (3, 3), padding='same', use_bias=False, name="conv_layer")
        self.batch_norm_layer = layers.BatchNormalization(
            name="batch_norm_layer")

      def call(self, x):
        y = tf.nn.conv2d(x, self.filter, (1, 1), 'SAME')
        y = self.batch_norm_layer(y)
        return self.conv_layer(y)

    model = SubclassedModel(name='subclassed')
    input_shape = (12, 12, 16)
    model.build(input_shape=(None,) + input_shape)

    expected_log = ['0/1 variables (Conv2D or MatMul) are permuted!']
    self._CheckPermuteLogs(model, expected_log, input_shape,
                           plot_to_file=inspect.currentframe().f_code.co_name)


  def testPermuteMixedApisWithEmptyInferlistOp(self):
    # A simple model of Conv2D->BatchNorm->Conv2D. Since the BatchNorm is not
    # from a keras layer, none of the layers would be permuted.
    class SubclassedModel(tf.keras.Model):
      def __init__(self, name):
        super(SubclassedModel, self).__init__(name=name)
        v_init = tf.random_normal_initializer()
        self.conv2d_1 = layers.Conv2D(
            32, (3, 3), padding='same', use_bias=False, name="conv2d_1")
        self.conv2d_2 = layers.Conv2D(
            32, (3, 3), padding='same', use_bias=False, name="conv2d_2")
        self.scale = tf.Variable(
            initial_value=v_init(shape=(32,), dtype='float32'), trainable=True)
        self.offset = tf.Variable(
            initial_value=v_init(shape=(32,), dtype='float32'), trainable=True)

      def call(self, x):
        y = self.conv2d_1(x)
        y, _, _ = tf.compat.v1.nn.fused_batch_norm(y, self.scale, self.offset)
        return self.conv2d_2(y)

    model = SubclassedModel(name='subclassed')
    input_shape = (12, 12, 16)
    model.build(input_shape=(None,) + input_shape)

    expected_log = ['0/2 variables (Conv2D or MatMul) are permuted!']
    self._CheckPermuteLogs(model, expected_log, input_shape,
                           plot_to_file=inspect.currentframe().f_code.co_name)

  def testPermuteMixedPrecision(self):
    class SubclassedModel(tf.keras.Model):
      def __init__(self, name):
        super(SubclassedModel, self).__init__(name=name)
        self.conv2d_1 = layers.Conv2D(
            32, (3, 3), padding='same', use_bias=False, name="conv2d_1")
        self.batch_norm_1 = layers.BatchNormalization(name="batch_norm_1")
        self.conv2d_2 = layers.Conv2D(
            32, (3, 3), padding='same', use_bias=False, name="conv2d_2")
        self.relu_1 = layers.ReLU(name='relu_1')

      def call(self, x):
        y = self.conv2d_1(x)
        y = self.batch_norm_1(y)
        y = self.relu_1(y)
        return self.conv2d_2(y)

    # When using the mixed precision policy, there will be many Cast ops
    # inserted after ReadVariableOp and we have patterns like
    # ReadVariableOp->Cast->Conv2D which we should treat them as skippable.
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model = SubclassedModel(name='subclassed')
    input_shape = (12, 12, 16)
    model.build(input_shape=(None,) + input_shape)

    expected_log = ['1/2 variables (Conv2D or MatMul) are permuted!']
    self._CheckPermuteLogs(model, expected_log, input_shape,
                           plot_to_file=inspect.currentframe().f_code.co_name)

    tf.keras.mixed_precision.set_global_policy('float32')


  def testPermuteMixedPrecisionOptimizerOrder(self):
    class SubclassedModel(tf.keras.Model):
      def __init__(self, name):
        super(SubclassedModel, self).__init__(name=name)
        self.conv2d_1 = layers.Conv2D(
            32, (3, 3), padding='same', use_bias=False, name="conv2d_1")
        self.batch_norm_1 = layers.BatchNormalization(name="batch_norm_1")
        self.conv2d_2 = layers.Conv2D(
            32, (3, 3), padding='same', use_bias=False, name="conv2d_2")
        self.relu_1 = layers.ReLU(name='relu_1')

      def call(self, x):
        y = self.conv2d_1(x)
        y = self.batch_norm_1(y)
        y = self.relu_1(y)
        return self.conv2d_2(y)

    # When using the mixed precision policy, there will be many Cast ops
    # inserted after ReadVariableOp and we have patterns like
    # ReadVariableOp->Cast->Conv2D which we should treat them as skippable.
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model = SubclassedModel(name='subclassed')
    input_shape = (12, 12, 16)
    model.build(input_shape=(None,) + input_shape)

    opt = optimizers.legacy.SGD(learning_rate=0.2, momentum=1.0)
    # The permute is triggered during the init stage of the ASP wrapper.
    opt = tf_asp.AspOptimizerWrapperV2(
              opt, model, permute=True, padding=True,
              search_device='GPU', input_shapes=(None,) + input_shape)

    # The LossScaleOptimizer needs to be the laster wrapper.
    # TODO(kaixih): If we need to relax this ordering requirement.
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=False,
                                                      initial_scale=128.0)
    model.compile(optimizer=opt, loss="mse")

    x_train = tf.random.normal(shape=(100, 12, 12, 16))
    y_train = tf.random.normal(shape=(100, 12, 12, 32))

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(10)

    model.fit(train_dataset, epochs=1, steps_per_epoch=1)


    tf.keras.mixed_precision.set_global_policy('float32')


  def testSameVariableNames(self):
    input_shape = (16, 16, 64)
    inner_model = tf.keras.Sequential(name='b')
    inner_model.add(layers.Conv2D(16, (3, 3), padding='same',
                                  input_shape=input_shape,
                                  name='b/c/conv2d_1'))
    model = tf.keras.Sequential(name='a')
    model.add(inner_model)
    model.add(layers.Conv2D(32, (3, 3), name='b/c/conv2d_1'))

    expected_logs = [
        "Permute-C: node_name=a/b/c/conv2d_1/Conv2D",
        "Permute-K: node_name=a/b/b/c/conv2d_1/BiasAdd",
        "Permute-K: node_name=a/b/b/c/conv2d_1/Conv2D"]
    self._CheckPermuteLogs(model, expected_logs, input_shape)


  def testSameOpNames(self):
    input_shape = (100, 100, 64)
    inner_model1 = tf.keras.Sequential(name='a/b')
    inner_model1.add(layers.Conv2D(64, (3, 3), padding='same',
                                  input_shape=input_shape,
                                  name='c/conv2d_1'))
    inner_model2 = tf.keras.Sequential(name='a')
    inner_model2.add(layers.Conv2D(16, (3, 3), padding='same',
                                  input_shape=input_shape,
                                  name='b/c/conv2d_1'))
    model = tf.keras.Sequential(name='n')
    model.add(inner_model1)
    model.add(inner_model2)

    expected_logs = [
      "Failed to distinguish variables for op_name=n/a/b/c/conv2d_1/Conv2D,",
      "Failed to distinguish variables for op_name=n/a/b/c/conv2d_1/Conv2D_1,",
      "Failed to distinguish variables for op_name=n/a/b/c/conv2d_1/BiasAdd,",
      "Failed to distinguish variables for op_name=n/a/b/c/conv2d_1/BiasAdd_1,",
    ]
    self._CheckPermuteLogs(model, expected_logs, input_shape)


  def testUnsupportedSavedModel(self):
    input_shape = (16, 16, 64)
    conv_1 = layers.Conv2D(16, (3, 3), padding='same', name='conv_1')
    conv_2 = layers.Conv2D(32, (3, 3), padding='same', name='conv_2')
    input1 = layers.Input(shape=input_shape)
    output1 = conv_1(input1)
    output1 = conv_2(output1)
    model = tf.keras.models.Model(inputs=input1, outputs=output1)


    expected_logs = [
        "Permute-C: node_name=model/conv_2/Conv2D",
        "Permute-K: node_name=model/conv_1/BiasAdd",
        "Permute-K: node_name=model/conv_1/Conv2D"]
    self._CheckPermuteLogs(model, expected_logs, input_shape)

    try:
      tmpdir = tempfile.mkdtemp()
      tf.saved_model.save(model, tmpdir)
      loaded = tf.saved_model.load(tmpdir)
    finally:
      shutil.rmtree(tmpdir)

    with self.assertRaisesRegex(
        ValueError, '`model` can only be a `tf.keras.Model` instance.'):
      self._CheckPermuteLogs(loaded, [], input_shape)

    infer = loaded.signatures['serving_default']

    with self.assertRaisesRegex(
        ValueError, '`model` can only be a `tf.keras.Model` instance.'):
      self._CheckPermuteLogs(infer, [], input_shape)

  def testLargeInputDim(self):
    model = tf.keras.Sequential(name="seq")
    input_shape = (512, )
    model.add(layers.Dense(512, input_shape=input_shape))
    model.add(layers.Dense(2064))
    model.add(layers.Dense(512))

    expected_logs = [
        "[TF-ASP] Finally swap"]
    self._CheckPermuteLogs(model, expected_logs, input_shape)

if __name__ == "__main__":
  test.main()
