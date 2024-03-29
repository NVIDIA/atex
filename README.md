# NVIDIA/Atex: A TensorFlow Extension

This repository holds NVIDIA-maintained utilities to improve GPU performance and
usability for Tensorflow training and inference. The intent of Atex is to make
up-to-date utilities available to users as quickly as possible. As such, these
utilities are experimental. Some we will upstream and support permanently in
projects such as TensorFlow or Keras. Others will eventually be discontinued.

## Contents

### 1. Fused Layer/Instance Normalization

`nv_norms.LayerNormalization` and `nv_norms.InstanceNormalization` are fused
implementations designed to replace the `tf.keras.layers.LayerNormalization` and
`tfa.layers.InstanceNormalization`. The Keras and Addons implementations compose
many ops, such as Mul, Add, etc., to compute the stats and then scale/offset the
results. In contrast, the NVIDIA fused implementation provided in `nv_norms`
compute the norms in a single operation, eliminating many expensive round-trips
to device memory and significantly improving performance. More in
[nv_norms](./atex/nv_norms/).

### 2. Structured Sparsity

This is a project for Tensorflow on supporting fine-grained structured sparsity
for the NVIDIA Ampere GPU architecture. We only need users to add a couple lines
to their python script and then the pretrained model can be automatically pruned
to benefit from the sparse Tensor Cores (available from Ampere GPUs) to achieve
faster inference speed after deployment. More in
[structured_sparsity](./atex/structured_sparsity/).


## Installation from source

To build this package from source, run the following command in the root directory of this package.

```
pip install .
```

## Contribution guidelines

Please review the [Contribution Guidelines](CONTRIBUTING.md). 

[GitHub issues](https://github.com/nvidia/atex/issues) will be used for tracking
requests and bugs.

## License

[BSD License 2.0](LICENSE)
