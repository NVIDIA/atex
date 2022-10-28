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
Successfully installed tensorflow-nv-norms-0.0.1
```

### Benchmark
Here is a simple benchmark script to compare the performance of
`tf.keras.layers.LayerNormalization` and our fused layer norm.
```bash
# python -u benchmark_layer_norm.py
# On V100
Input: 10 10000000 Time(ms): TF: 28.10 NV: 5.56
Input: 100 1000000 Time(ms): TF: 7.97 NV: 4.39
Input: 1000 100000 Time(ms): TF: 7.63 NV: 3.60
Input: 10000 10000 Time(ms): TF: 8.00 NV: 4.90
Input: 100000 1000 Time(ms): TF: 11.16 NV: 4.39
Input: 1000000 100 Time(ms): TF: 30.57 NV: 6.73
Input: 10000000 10 Time(ms): TF: 241.46 NV: 14.00
Input: 4 400001 Time(ms): TF: 2.19 NV: 0.65
Input: 4 10000001 Time(ms): TF: 23.33 NV: 2.60

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
```
Here is a simple benchmark script to compare the performance of
`tfa.layers.InstanceNormalization` and our fused instance norm.
```bash
# python -u benchmark_instance_norm.py
# On V100
Input: (2, 6, 6, 6, 32) Time(ms): TF: 3.75 NV: 0.56
Input: (2, 128, 128, 128, 32) Time(ms): TF: 36.45 NV: 9.35
Input: (2, 128, 128, 128, 64) Time(ms): TF: 71.64 NV: 16.07
Input: (4, 128, 128, 128, 32) Time(ms): TF: 71.39 NV: 16.03
Input: (4, 64, 64, 64, 64) Time(ms): TF: 18.77 NV: 6.11
Input: (8, 64, 64, 64, 32) Time(ms): TF: 18.78 NV: 6.08
Input: (8, 64, 64, 64, 64) Time(ms): TF: 36.42 NV: 9.39
Input: (8, 64, 64, 64, 128) Time(ms): TF: 71.43 NV: 16.08
Input: (4, 32, 32, 32, 256) Time(ms): TF: 10.20 NV: 4.04
Input: (8, 32, 32, 32, 256) Time(ms): TF: 18.51 NV: 5.71
End of channel last layout.
Input: (2, 32, 6, 6, 6) Time(ms): TF: 3.67 NV: 0.55
Input: (2, 32, 128, 128, 128) Time(ms): TF: 34.55 NV: 6.83
Input: (2, 64, 128, 128, 128) Time(ms): TF: 68.09 NV: 13.63
Input: (4, 32, 128, 128, 128) Time(ms): TF: 68.09 NV: 12.65
Input: (4, 64, 64, 64, 64) Time(ms): TF: 17.69 NV: 3.73
Input: (8, 32, 64, 64, 64) Time(ms): TF: 17.67 NV: 3.73
Input: (8, 64, 64, 64, 64) Time(ms): TF: 34.50 NV: 6.75
Input: (8, 128, 64, 64, 64) Time(ms): TF: 68.29 NV: 13.05
Input: (4, 256, 32, 32, 32) Time(ms): TF: 10.04 NV: 1.79
Input: (8, 256, 32, 32, 32) Time(ms): TF: 17.89 NV: 3.44
End of channel first layout.
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
