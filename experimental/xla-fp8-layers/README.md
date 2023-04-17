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
modell as shown in the example below.

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
$ python transformer.py # fp32 84ms
$ python transformer.py --mixed # fp16 45ms
$ python transformer.py --mixed --fp8 # fp16+fp8 35ms
```

## A JAX Example with High-Level API

WIP

## More on FP8

The high-level APIs (shown above) hides the complexity of handling FP8, which
includes:
* All FP8-safe operations have their inputs cast to FP8
* Amax history is updated
* New scaling factors are computed and ready for the next iteration

If users want maximum flexibility and control over how and where to use the FP8
operations. Please refer to this [FP8 tutorial on low-level FP8
API](./fp8-tutorial.md)



