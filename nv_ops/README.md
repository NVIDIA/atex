# NVIDIA Custom Ops

This repo contains fused layer/instance normalization op for Tensorflow. It is
designed to replace the `tf.keras.layers.LayerNormalization` and 
`tfa.layers.InstanceNormalization` which consist of many ops, such as Mul, Add, 
etc. Instead, we try to fuse these ops to save the device memory round trip and 
improve the performance.

## Expected Performance

Typically, the normalization can be described in two parts: (1) Computing the
mean/variance and normalizing the input; (2) Scaling and offsetting the output
from (1). `tf.keras.layers.LayerNormalization` relies on the CuDNN
BatchNormalization for the first part but not for the second part, since the
shapes of mean/variance and gamma/beta are different between layer and batch
normalization. On the other hand, `tfa.layers.InstanceNormalization` utilizes
many operations, like `Mul`, `Mean`, `Add`, etc. Therefore, by using `nv_norms`,
users should expect more performance benefits from layer normalization than
instance normalization.

In addition, `nv_norms` is compatible with the TF grappler optimizations (e.g.
layout). In the NGC TF, the layout optimizer has been improved to recognize the
newly introduced ops (`FusedLayerNorm`, `FusedLayerNormGrad`,
`FusedInstanceNorm` and `FusedInstanceNormGrad`) to avoid unnecessary Transpose
ops when operating with the `float` or `half` data types. 

## Usage

### Install
We are going to build the `nv_norms` inside a NGC Tensorflow Docker container.
We create a bash script `setup.sh` to facilitate the installation and a sanity
check. Then, the built PIP package is located in `artifacts/` and we can install
it as:

```bash
# bash setup.sh
<...> === Output wheel file is in: <...>/nv_ops/artifacts

# pip install artifacts/*.whl
<...>
Successfully installed tensorflow-nv-norms-0.0.2
```

### Benchmark
Here is a simple benchmark script to compare the performance of
`tf.keras.layers.LayerNormalization` and our fused layer norm.
```bash
$ python -u benchmark_layer_norm.py

# On A100
Input: 10 10000000 Time(ms): TF: 28.97 NV: 3.36
Input: 100 1000000 Time(ms): TF: 5.69 NV: 2.78
Input: 1000 100000 Time(ms): TF: 5.48 NV: 2.81
Input: 10000 10000 Time(ms): TF: 6.35 NV: 4.50
Input: 100000 1000 Time(ms): TF: 9.48 NV: 4.32
Input: 1000000 100 Time(ms): TF: 23.04 NV: 6.33
Input: 10000000 10 Time(ms): TF: 178.51 NV: 13.82
Input: 4 400001 Time(ms): TF: 2.30 NV: 0.59
Input: 4 10000001 Time(ms): TF: 26.08 NV: 1.69

# On H100
Input: 10 10000000 Time(ms): TF: 16.91 NV: 2.36
Input: 100 1000000 Time(ms): TF: 4.28 NV: 1.78
Input: 1000 100000 Time(ms): TF: 4.22 NV: 1.95
Input: 10000 10000 Time(ms): TF: 5.26 NV: 5.84
Input: 100000 1000 Time(ms): TF: 7.89 NV: 4.73
Input: 1000000 100 Time(ms): TF: 20.04 NV: 6.26
Input: 10000000 10 Time(ms): TF: 160.06 NV: 10.61
Input: 4 400001 Time(ms): TF: 3.10 NV: 0.74
Input: 4 10000001 Time(ms): TF: 23.51 NV: 1.15
```
Here is a simple benchmark script to compare the performance of
`tfa.layers.InstanceNormalization` and our fused instance norm.
```bash
$ python -u benchmark_instance_norm.py

# On A100 (channels_last)
Input: (2, 6, 6, 6, 32) Time(ms): TF: 7.14 NV: 0.78
Input: (2, 128, 128, 128, 32) Time(ms): TF: 18.56 NV: 7.51
Input: (2, 128, 128, 128, 64) Time(ms): TF: 36.93 NV: 12.53
Input: (4, 128, 128, 128, 32) Time(ms): TF: 37.24 NV: 12.35
Input: (4, 64, 64, 64, 64) Time(ms): TF: 9.59 NV: 5.36
Input: (8, 64, 64, 64, 32) Time(ms): TF: 9.78 NV: 5.35
Input: (8, 64, 64, 64, 64) Time(ms): TF: 19.04 NV: 7.65
Input: (8, 64, 64, 64, 128) Time(ms): TF: 38.04 NV: 12.33
Input: (4, 32, 32, 32, 256) Time(ms): TF: 7.27 NV: 3.76
Input: (8, 32, 32, 32, 256) Time(ms): TF: 10.11 NV: 5.01

# On A100 (channels_first)
Input: (2, 32, 6, 6, 6) Time(ms): TF: 7.11 NV: 0.77
Input: (2, 32, 128, 128, 128) Time(ms): TF: 17.20 NV: 5.20
Input: (2, 64, 128, 128, 128) Time(ms): TF: 33.70 NV: 10.53
Input: (4, 32, 128, 128, 128) Time(ms): TF: 34.20 NV: 9.27
Input: (4, 64, 64, 64, 64) Time(ms): TF: 8.80 NV: 2.98
Input: (8, 32, 64, 64, 64) Time(ms): TF: 8.99 NV: 3.01
Input: (8, 64, 64, 64, 64) Time(ms): TF: 17.21 NV: 5.32
Input: (8, 128, 64, 64, 64) Time(ms): TF: 34.16 NV: 10.37
Input: (4, 256, 32, 32, 32) Time(ms): TF: 7.41 NV: 1.18
Input: (8, 256, 32, 32, 32) Time(ms): TF: 8.77 NV: 2.57

# On H100 (channels_last)
Input: (2, 6, 6, 6, 32) Time(ms): TF: 6.20 NV: 0.70
Input: (2, 128, 128, 128, 32) Time(ms): TF: 14.86 NV: 6.05
Input: (2, 128, 128, 128, 64) Time(ms): TF: 28.65 NV: 8.69
Input: (4, 128, 128, 128, 32) Time(ms): TF: 28.65 NV: 8.70
Input: (4, 64, 64, 64, 64) Time(ms): TF: 7.81 NV: 4.63
Input: (8, 64, 64, 64, 32) Time(ms): TF: 7.73 NV: 4.66
Input: (8, 64, 64, 64, 64) Time(ms): TF: 14.86 NV: 5.98
Input: (8, 64, 64, 64, 128) Time(ms): TF: 28.65 NV: 8.63
Input: (4, 32, 32, 32, 256) Time(ms): TF: 6.27 NV: 3.57
Input: (8, 32, 32, 32, 256) Time(ms): TF: 7.87 NV: 4.25

# On H100 (channels_first)
Input: (2, 32, 6, 6, 6) Time(ms): TF: 6.17 NV: 0.69
Input: (2, 32, 128, 128, 128) Time(ms): TF: 14.14 NV: 3.85
Input: (2, 64, 128, 128, 128) Time(ms): TF: 26.81 NV: 7.57
Input: (4, 32, 128, 128, 128) Time(ms): TF: 26.81 NV: 6.16
Input: (4, 64, 64, 64, 64) Time(ms): TF: 7.14 NV: 2.17
Input: (8, 32, 64, 64, 64) Time(ms): TF: 7.14 NV: 2.17
Input: (8, 64, 64, 64, 64) Time(ms): TF: 13.86 NV: 3.51
Input: (8, 128, 64, 64, 64) Time(ms): TF: 26.70 NV: 6.38
Input: (4, 256, 32, 32, 32) Time(ms): TF: 6.19 NV: 0.91
Input: (8, 256, 32, 32, 32) Time(ms): TF: 7.13 NV: 1.71
```

### Use it in Real-World Model
We provide the sample scripts to demonstrate how to substitute the
`nv_norms.XXXNormalization` for the existing layer calls.

To replace `tf.keras.layers.LayerNormalization` (details in `sample_layerN.py`):
```python
layerN = tf.keras.layers.LayerNormalization(axis=(1,2,3))
```
To
```python
layerN = nv_norms.LayerNormalization(axis=(1,2,3))
```

To replace `tfa.layers.InstanceNormalization` (details in
`sample_instanceN.py`):
```python
instanceN = tfa.layers.InstanceNormalization(axis=channel_axis)
```
To
```python
instanceN = nv_norms.InstanceNormalization(axis=channel_axis)
```
A legal value of optional argument `axis` is taken from (1, -1), where -1 is the
 default.

### Limitations

* The `axis` argument can accept a list/tuple of integers. Typically this is the
  features axis/axes. The left-out axes are typically the batch axis/axes. This
  argument defaults to `[-1]`. We only support a list of packed axes that must
  include the last dimension, e.g., `[-3, -2, -1]` but not `[-3, -1]`. 

### Update Notes:
* v0.0.2
  * Improved the bandwidth usage of the instance normalization via
    vectorization.
  * Fixed a potential data race issue in the instance normalization kernels.
  * Updated the code to be built with C++17.
  * Updated the performance numbers on H100 GPUs.
