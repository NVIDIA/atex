# NVIDIA Custom Ops

This repo contains fused layer/instance normalization op for Tensorflow. It is
designed to replace the `tf.keras.layers.LayerNormalization` and 
`tfa.layers.InstanceNormalization` which consist of many ops, such as Mul, Add, 
etc. Instead, we try to fuse these ops to save the device memory round trip and 
improve the performance.

## Expected Performance

Typically, the normalization can be described as two parts: (1) computing the
mean/variance and normalizing the input; (2) scaling and offsetting the output
of (1). The `tf.keras.layers.LayerNormalization` has already used the CuDNN for
the first part but not the second part due to some dimension restrictions of
CuDNN. In comparison, our implementation can achieve up to 18x speedups. A
detailed performance analysis can be found
[here](https://docs.google.com/spreadsheets/d/1KM3VlGL3GqjV_o7iSKHqBHhkry6Ha0OmYKtHl1_I7wA/edit?usp=sharing).

The fp16 support should be compatible with the current TF grappler optimizations
(e.g. layout). However, we need an additional patch to let the layout optimizer
recognize the newly introduced ops (`FusedLayerNorm`, `FusedLayerNormGrad`,
`FusedInstanceNorm` and `FusedInstanceNormGrad`) to avoid unnecessary Transpose 
ops. This will be done in the future NGC TF release. 


## Usage

### Install
We are going to build the `nv_norms` inside a NGC Tensorflow Docker container.
We create a bash script `setup.sh` to facilitate the installation and a sanity
check. Then, the built PIP package is located in `artifacts/` and we can install
it as:

```bash
# bash setup.sh
<...>
Ran 5 tests in 4.131s

OK (skipped=1)
# pip install artifacts/*.whl
<...>
Successfully installed tensorflow-nv-norms-0.0.1
```

### Benchmark
Here is a simple benchmark script to compare the performance of
`tf.keras.layers.LayerNormalization` and our fused layer norm.
```bash
# python -u benchmark_layer_norm.py
Input: 10 10000000 Time(ms): TF: 26.56 NV: 6.69
Input: 100 1000000 Time(ms): TF: 8.33 NV: 6.02
Input: 1000 100000 Time(ms): TF: 8.09 NV: 5.15
Input: 10000 10000 Time(ms): TF: 8.56 NV: 6.48
Input: 100000 1000 Time(ms): TF: 11.71 NV: 6.37
Input: 1000000 100 Time(ms): TF: 31.85 NV: 8.75
Input: 10000000 10 Time(ms): TF: 249.39 NV: 14.02
```
Here is a simple benchmark script to compare the performance of
`tfa.layers.InstanceNormalization` and our fused instance norm.
```bash
# python -u benchmark_instance_norm.py
Input: (2, 128, 128, 128, 32) Time(ms): TF: 36.60 NV: 10.01
Input: (2, 128, 128, 128, 64) Time(ms): TF: 71.79 NV: 17.26
Input: (4, 128, 128, 128, 32) Time(ms): TF: 71.77 NV: 17.23
Input: (4, 64, 64, 64, 64) Time(ms): TF: 18.86 NV: 6.52
Input: (8, 64, 64, 64, 32) Time(ms): TF: 18.84 NV: 6.50
Input: (8, 64, 64, 64, 64) Time(ms): TF: 36.54 NV: 10.01
Input: (8, 64, 64, 64, 128) Time(ms): TF: 71.76 NV: 17.16
Input: (4, 32, 32, 32, 256) Time(ms): TF: 10.23 NV: 4.25
Input: (8, 32, 32, 32, 256) Time(ms): TF: 18.78 NV: 6.14
End of NHWC
Input: (2, 32, 128, 128, 128) Time(ms): TF: 34.85 NV: 10.19
Input: (2, 64, 128, 128, 128) Time(ms): TF: 68.45 NV: 20.06
Input: (4, 32, 128, 128, 128) Time(ms): TF: 68.45 NV: 16.79
Input: (4, 64, 64, 64, 64) Time(ms): TF: 17.96 NV: 5.67
Input: (8, 32, 64, 64, 64) Time(ms): TF: 17.94 NV: 5.66
Input: (8, 64, 64, 64, 64) Time(ms): TF: 34.78 NV: 9.14
Input: (8, 128, 64, 64, 64) Time(ms): TF: 68.44 NV: 16.08
Input: (4, 256, 32, 32, 32) Time(ms): TF: 10.12 NV: 2.11
Input: (8, 256, 32, 32, 32) Time(ms): TF: 17.84 NV: 4.06
End of NCHW
```

### Use it in Real-World Model
We provide the sample scripts to demonstrate how to substitute the
`nv_norms.fused_xxx_norm` for the existing layer calls.

To replace `tf.keras.layers.LayerNormalization` (details in `sample_layerN.py`):
```python
layerN = tf.keras.layers.LayerNormalization(axis=(1,2,3))
...
z = layerN(y)
```
To
```python
layerN = tf.keras.layers.LayerNormalization(axis=(1,2,3))
...
z, _, _ = nv_norms.fused_layer_norm(y, layerN.gamma, layerN.beta)
```

To replace `tfa.layers.InstanceNormalization` (details in
`sample_instanceN.py`):
```python
instanceN = tfa.layers.InstanceNormalization(axis=channel_axis)
...
y = instanceN(x)
```
To
```python
y, _, _ = nv_norms.fused_instance_norm(
    x, instanceN.weights[0], instanceN.weights[1], data_format='N...C')
```
A legal value of optional argument `data_format` is taken from ("N...C", "NC...", "NCHW", "NHWC", "NDHWC", "NCDHW"),
where `'NHWC'` is the default.
