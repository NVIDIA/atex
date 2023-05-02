# FP8 Custom Layers

**NOTE** This is a project for TF/JAX on supporting fp8 computations via XLA for
the NVIDIA Hopper GPU architecture. It is still a WIP implementation that has
not been extensively tested yet.

> **NOTE** The 8-bit floating-point (FP8) data types are only featured on NVIDIA
> Hopper GPUs or newer (>= SM90) and require CUDA 11.8 or higher.

This repo provides a collection of custom fully-connected layers in TF and JAX
to help utilize the 8-bit floating point (FP8) precision on Hopper GPUs for
better performance with lower memory utilization in both training and inference.

There are multiple ways to take advantage of FP8, such as the custom FP8 kernel
based method (e.g., [Transformer
Engine](https://github.com/NVIDIA/TransformerEngine)) or the native-XLA
compiler-based method, which is the subject of this repo.

To help the use of XLA-FP8, in this repo, we provide two high-level APIs for
using FP8, namely the `fp8layers.tensorflow.Dense` and
`fp8layers.jax.DenseGeneral` layers which are a drop-in replacement for
`keras.layers.Dense` (TensorFlow) and `flax.linen.DenseGeneral` (JAX)
respectively. You can directly place them into your XLA JIT-compiled functions
to carry out computation in FP8.

## Installation

To install the package, users can execute the following command in the root
directory that contains the `setup.py`.

```bash
$ pip install .
<...>
Successfully installed fp8layers-python-0.1.0
```
## A TensorFlow Example with High-Level API

Using XLA-FP8 in TF is simple and users only need to replace the
`tf.keras.layers.Dense` with `fp8layers.tensorflow.Dense` in the JIT-compiled
model as shown in the example below.

```python
from fp8layers.tensorflow import Dense

@tf.function(jit_compile=True)
def train_step(x):
  #y = tf.keras.layers.Dense(16)(x)
  y = Dense(16)(x)
```

If you do not JIT your functions, you will also need to set an extra environment
variable to invoke autojit.

```bash
TF_XLA_FLAGS="--tf_xla_auto_jit=2" 
```

We have also provided an example of using the high-level API in the basic
transformer layer, a GPT encoder adapted from
[here](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/quickstart.html).

```bash
$ python examples/tensorflow/transformer.py # fp32 84ms
$ python examples/tensorflow/transformer.py --mixed # fp16 45ms
$ python examples/tensorflow/transformer.py --mixed --fp8 # fp16+fp8 35ms
```

## A JAX Example with High-Level API

Using XLA-FP8 in JAX is as easy as the TF and users only need to replace the
`flax.linen.Dense[General]` with `fp8layers.jax.DenseGeneral` in the
JIT-compiled model as shown in the example below.

```python
from fp8layers.jax import DenseGeneral

class Foo(nn.Module):
  @nn.compact
  def __call__(self, x):
    #y = flax.linen.DenseGeneral(16)(x)
    y = DenseGeneral(16)(x)
foo = Foo()
...
fn = jax.jit(foo.apply)
```

Similarly, users can try out the basic encoder examples:

```bash
$ python examples/jax/transformer.py # fp32 79ms
$ python examples/jax/transformer.py --mixed # bf16 45ms
$ python examples/jax/transformer.py --mixed --fp8 # bf16+fp8 32ms
```

### Updating the FP8 Parameters
Note, FLAX is a functional framework, meaning the layers are stateless and the
parameters (such as the kernel and bias) are stored outside them. To follow this
convention, we put the fp8-related parameters (i.e., scales and amax history)
under `fp8_params` collection and won't update them during the train step. This
is different from what we've done in the TF. In JAX, we return the new
parameters as their "grads" and users need to replace the old parameters. We
also provide a helper `TrainState` class to help with it. The pseudocode below
shows the usage.

```python
# Manually update the parameters:
loss_val, grads = train_step_fn(variables, ...)
# Update the fp8_params by simply relacing the old ones.
variables['fp8_params'] = grads['fp8_params']
# Update the non-fp8 params by using the existing optimizer.
updates = opt.update(variables['params'], ...)
variables['params'] = optax.apply_updates(updates)

# ... Or use our provided TrainState:
train_state = TrainState.create(params=variables, ...)
loss_val, grads = train_step_fn(train_state.params, ...)
train_state = train_state.apply_gradients(grads=grads)
```

### Supporting Multi-GPUs
The provided `DenseGeneral` supposes the multiple GPUs and allow users to
specify the "logical axis annotations" for the parameters (i.e., `kernel_axes`
and `bias_axes`). If specified, the parameters will be partitioned accordingly
before the matrix multiplications. Correspondingly, the amax computation will be
conducted locally on each device's shard and then the collective all-reduce is
used to get the actual amax. This global amax will be used in computing the new
scaling factor and updating the amax history. All these partition and collective
operations are done automatically by XLA. [These unit
tests](./tests/jax/test_partition.py) provide a plenty of samples of various
sharding strategies.


## More on FP8

The high-level APIs (shown above) hides the complexity of handling FP8, which
includes:
* All FP8-safe operations have their inputs cast to FP8
* Amax history is updated
* New scaling factors are computed and ready for the next iteration

If users want maximum flexibility and control over how and where to use the FP8
operations. Please refer to this [FP8 tutorial on low-level FP8
API](./fp8-tutorial.md)



