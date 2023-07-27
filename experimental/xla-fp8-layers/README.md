# FP8 Custom Layers

> **NOTE** The 8-bit floating-point (FP8) data types are only featured on NVIDIA
> Hopper GPUs or newer (>= SM90) and require CUDA 11.8 or higher.

This repo provides a collection of custom fully-connected layers in TF and JAX
(incl. FLAX and Praxis) to help utilize the 8-bit floating point (FP8) precision
on Hopper GPUs for better performance with lower memory utilization in both
training and inference.

There are multiple ways to take advantage of FP8, such as the custom FP8 kernel
based method (e.g., [Transformer
Engine](https://github.com/NVIDIA/TransformerEngine)) or the native-XLA
compiler-based method, which is the subject of this repo.

To help the use of XLA-FP8, we provide high-level APIs as a drop-in replacement
option for different frameworks:

* TF: `keras.layers.Dense` => `fp8layers.tensorflow.Dense` 
* FLAX: `flax.linen.DenseGeneral` => `fp8layers.flax.DenseGeneral`
* Praxis: `praxis.layers.Linear` => `fp8layers.praxis.Linear`

Users can directly place them into your XLA JIT-compiled functions to carry out
computation in FP8.

## Installation

To install the package, users can execute the following command in the root
directory that contains the `setup.py`.

```bash
$ pip install .
<...>
Successfully installed fp8layers-python-0.1.0
```

### Recommended Containers (Internal-only)

* [TF](https://gitlab-master.nvidia.com:5005/dl/dgx/tensorflow:master-py3-devel)
* [FLAX/Praxis](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/pax)

### Known issues of compatibility
* Flax version should be no less than 0.6.9 on which JAX dense layer
  implementation depends.

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
$ python examples/tensorflow/transformer.py # fp32 48ms
$ python examples/tensorflow/transformer.py --mixed # fp16 23ms
$ python examples/tensorflow/transformer.py --mixed --fp8 # fp16+fp8 20ms
```

## A FLAX Example with High-Level API

Using XLA-FP8 in FLAX is as easy as the TF and users only need to replace the
`flax.linen.Dense[General]` by `fp8layers.flax.DenseGeneral` in the JIT-compile
model as shown in the example below. Note that the fp8 parameters are defined in
`fp8_params` collection.

```python
from fp8layers.flax import DenseGeneral

class Foo(nn.Module):
  @nn.compact
  def __call__(self, x):
    #y = flax.linen.DenseGeneral(16)(x)
    y = DenseGeneral(16)(x)
foo = Foo()
...
fn = jax.jit(foo.apply)
...
y = fn(variables, x)
```

Similarly, users can try out the basic encoder examples:

```bash
$ python examples/flax/transformer.py # fp32 79ms
$ python examples/flax/transformer.py --mixed # bf16 45ms
$ python examples/flax/transformer.py --mixed --fp8 # bf16+fp8 32ms
```

## A Praxis Example with High-Level API

Praxis provides another way to abstract the linear transformation based on
FLAX. Accordingly, users can use `fp8layers.praxis.Linear` in place of
`praxis.layers.Linear`.

```python
from fp8layers.praxis import Linear

class Foo(base_layer.BaseLayer):
  ...
  #linear_tpl: LayerTpl = template_field(praxis.layers.Linear)
  linear_tpl: LayerTpl = template_field(fp8layers.praxis.Linear)
  ...
foo = Foo()
...
fn = jax.jit(foo.apply)
```


### Updating the FP8 Parameters
Note, FLAX is a functional framework, meaning the layers are stateless and the
parameters (such as the kernel and bias) are stored outside them. To follow this
convention, we store the fp8-related parameters (i.e., scales and amax history)
under the `fp8_params` collection as
`flax.linen.partitioning.variable_with_axes`. Note, the new values of the fp8
parameters are returned as their "grads" defined in `custom_vjp`. So, a typical
variable update is like (1) for non-fp8 variables, we use the grads in the given
optimizer as usual; (2) for the fp8 variables, we replace the variables with
their grads. In the repo, we provide a `TrainState` class to show how we should
do it.


### Supporting Multi-GPUs
The provided `DenseGeneral` supposes the multiple GPUs and allow users to
specify the "logical axis annotations" for the parameters (i.e., `kernel_axes`
and `bias_axes`). If specified, the parameters will be partitioned accordingly
before the matrix multiplications. Correspondingly, the amax computation will be
conducted locally on each device's shard and then the collective all-reduce is
used to get the actual amax. This global amax will be used in computing the new
scaling factor and updating the amax history. All these partition and collective
operations are done automatically by XLA. [These unit
tests](./tests/flax/test_partition.py) provide a plenty of samples of various
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



