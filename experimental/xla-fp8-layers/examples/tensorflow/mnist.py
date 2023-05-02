"""Runs a simple mnist model with FP8. """
import argparse
import time

import tensorflow as tf
from keras import layers

from fp8layers.tensorflow import Dense

tf.keras.utils.set_random_seed(123)

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--fp8', action='store_true', help='use_fp8')
parser.add_argument('--mixed', action='store_true',
                    help='Enable mixed precision and fp16 compute type')
args = parser.parse_args()

use_fp8 = args.fp8
use_mixed = args.mixed

if use_mixed:
  tf.keras.mixed_precision.set_global_policy('mixed_float16')

DenseLayer = Dense if use_fp8 else layers.Dense

class MnistModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = DenseLayer(64, activation="relu")
    self.dense2 = DenseLayer(64, activation="relu")
    self.dense3 = DenseLayer(10)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    output = self.dense3(x)
    return output

model = MnistModel()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

batch_size = 64

# Prepare the trainging dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(batch_size)

# Prepare the metrics.
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function(jit_compile=True)
def train_step(x, y):
  # Open a GradientTape to record the operations run during the forward pass,
  # which enables auto-differentiation.
  with tf.GradientTape() as tape:

    # Run the forward pass of the layer. The operations that the layer applies
    # to its inputs are going to be recorded on the GradientTape.
    logits = model(x, training=True)  # Logits for this minibatch

    # Compute the loss value for this minibatch.
    loss_value = loss_fn(y, logits)

  # Use the gradient tape to automatically retrieve the gradients of the
  # trainable variables with respect to the loss.
  grads = tape.gradient(loss_value, model.trainable_weights)

  # Run one step of gradient descent by updating the value of the variables to
  # minimize the loss.
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  train_acc_metric.update_state(y, logits)
  return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

epochs = 20
start_time = time.time()
for epoch in range(epochs):
  print("\nStart of epoch %d" % (epoch,))
  # Iterate over the batches of the dataset.
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    loss_value = train_step(x_batch_train, y_batch_train)
    # Log every 200 batches.
    if step % 200 == 0:
      print("Training loss at step %d: %.4f" % (step, float(loss_value)))

  # Display metrics at the end of each epoch.
  train_acc = train_acc_metric.result()
  print("Training accuracy over epoch: %.4f" % (float(train_acc),))

  # Reset training metrics at the end of each epoch
  train_acc_metric.reset_states()

  # Run a validation loop at the end of each epoch.
  for x_batch_val, y_batch_val in val_dataset:
      test_step(x_batch_val, y_batch_val)

  val_acc = val_acc_metric.result()
  val_acc_metric.reset_states()
  print("Validation accuracy: %.4f" % (float(val_acc),))
print("Time taken: %.2fs" % (time.time() - start_time))
