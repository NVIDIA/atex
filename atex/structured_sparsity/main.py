# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


# WARNING:tensorflow:[TF-ASP] Allowlist is used: (Dense, Conv2D, )
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: dense_2 (type=Dense, shape=(128, 8))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_2 (type=Conv2D, shape=(3, 3, 16, 32))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_3 (type=Conv2D, shape=(3, 3, 32, 64))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_4 (type=Conv2D, shape=(3, 3, 64, 128))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_5 (type=Conv2D, shape=(3, 3, 128, 128))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_6 (type=Conv2D, shape=(3, 3, 128, 128))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_7 (type=Conv2D, shape=(3, 3, 128, 128))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_8 (type=Conv2D, shape=(3, 3, 128, 128))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_9 (type=Conv2D, shape=(3, 3, 128, 128))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_10 (type=Conv2D, shape=(7, 7, 128, 32))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_11 (type=Conv2D, shape=(7, 7, 32, 16))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: conv2d_12 (type=Conv2D, shape=(7, 7, 16, 8))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: dense (type=Dense, shape=(69192, 200))
# WARNING:tensorflow:[TF-ASP] Pruning list accepts the "kernel" variable from layer: dense_1 (type=Dense, shape=(200, 128))
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 222, 222, 8)       224
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 220, 220, 16)      1168
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 218, 218, 32)      4640
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 216, 216, 64)      18496
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 216, 216, 128)     73856
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 216, 216, 128)     147584
# _________________________________________________________________
# conv2d_6 (Conv2D)            (None, 216, 216, 128)     147584
# _________________________________________________________________
# conv2d_7 (Conv2D)            (None, 216, 216, 128)     147584
# _________________________________________________________________
# conv2d_8 (Conv2D)            (None, 216, 216, 128)     147584
# _________________________________________________________________
# conv2d_9 (Conv2D)            (None, 216, 216, 128)     147584
# _________________________________________________________________
# conv2d_10 (Conv2D)           (None, 210, 210, 32)      200736
# _________________________________________________________________
# conv2d_11 (Conv2D)           (None, 204, 204, 16)      25104
# _________________________________________________________________
# conv2d_12 (Conv2D)           (None, 198, 198, 8)       6280
# _________________________________________________________________
# conv2d_13 (Conv2D)           (None, 192, 192, 4)       1572
# _________________________________________________________________
# conv2d_14 (Conv2D)           (None, 186, 186, 2)       394
# _________________________________________________________________
# flatten (Flatten)            (None, 69192)             0
# _________________________________________________________________
# dense (Dense)                (None, 200)               13838600
# _________________________________________________________________
# dense_1 (Dense)              (None, 128)               25728
# _________________________________________________________________
# dense_2 (Dense)              (None, 8)                 1032
# =================================================================
# Total params: 14,935,750
# Trainable params: 14,935,750
# Non-trainable params: 0
# _________________________________________________________________

# signature_def['serving_default']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['conv2d_input'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 224, 224, 3)
#         name: serving_default_conv2d_input:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['dense_2'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 8)
#         name: StatefulPartitionedCall:0
#   Method name is: tensorflow/serving/predict


import tensorflow as tf
from tensorflow.keras import layers, optimizers, models

# ASP Step 1: Import the sparsity optimizer
from atex.structured_sparsity import tf_asp

tf.get_logger().setLevel(tf_asp.SHOW_PERMUTATION_INFO)

model = tf.keras.Sequential()
# Filter (2, 2, 4, 8): Skip pruning: input dim.
model.add(layers.Conv2D(8, (3, 3), padding='valid', activation="relu", input_shape=(224, 224, 3)))
# Filter (2, 2, 8, 16): Skip pruning: input dim.
model.add(layers.Conv2D(16, (3, 3), padding='valid', activation="relu"))
model.add(layers.Conv2D(32, (3, 3), padding='valid', activation="relu"))
model.add(layers.Conv2D(64, (3, 3), padding='valid', activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(layers.Conv2D(32, (7, 7), padding='valid', activation="relu"))
model.add(layers.Conv2D(16, (7, 7), padding='valid', activation="relu"))
model.add(layers.Conv2D(8, (7, 7), padding='valid', activation="relu"))
# Filter (7, 7, 8, 4): Skip pruning: input/output dim.
model.add(layers.Conv2D(4, (7, 7), padding='valid', activation="relu"))
# Filter (7, 7, 4, 2): Skip pruning: input/output dim.
model.add(layers.Conv2D(2, (7, 7), padding='valid', activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
# Filter (128, 8): Skip pruning: output dim.
model.add(layers.Dense(8, activation="sigmoid"))

model.summary()
#print("Init variables:", model.variables)

x = tf.random.uniform(shape=(32, 224, 224, 3))

opt = optimizers.legacy.SGD(learning_rate=0.2, momentum=1.0)

# ASP Step 2: Use AspOptimizerWrapper to wrap the existing optimizer.
opt = tf_asp.AspOptimizerWrapperV2(opt, model, padding=True,
                                   plot_to_file='main.png')

@tf.function
def train_step(x):
  with tf.GradientTape(persistent=True) as tape:
    y = model(x)
    loss = tf.reduce_mean(y)

  grads = tape.gradient(loss, model.variables)
  opt.apply_gradients(zip(grads, model.variables))
  return loss

for i in range(3):
  loss = train_step(x)
#print("Updated variables (masked):", model.variables)

export_savedmodel = True
if export_savedmodel:
  save_format = "exported_model"
  model.save(save_format)
  print(f"The model is saved to {save_format}")

  new_model = models.load_model(save_format)
  new_model.summary()
  result_checked = True
  for ref, new in zip(model.variables, new_model.variables):
    checked = tf.math.reduce_all(tf.math.equal(ref, new))
    if not checked:
      #print("Issue with:", new)
      result_checked = False
  print("Loaded Model checking:", "Passed" if result_checked else "Failed")

tf_asp.check_pruned_layers(model)

