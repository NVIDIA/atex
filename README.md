# NVIDIA/Atex: A TensorFlow Extension

This repository holds NVIDIA-maintained utilities to streamline faster training
and inference in TensorFlow. Some of the code here will be included in upstream
TensorFlow eventually. The intent of Atex is to make up-to-date utilities
available to users as quickly as possible.

## Contents

### 1. Fused Layer/Instance Normalization

`nv_norms.fused_layer_norm` and `nv_norms.fused_instance_norm` are fused
implementation designed to replace the `tf.keras.layers.LayerNormalization` and
`tfa.layers.InstanceNormalization`. The current backend of these normalization
ops essentially includes a set of many ops, such as Mul, Add, etc. to compute
the stats and then scale/offset the results. Instead, the fusion implementation
we used in `nv_norms` can reduce the expensive device memory round-trip and
conduct the computation more efficiently. More in [nv_ops](./nv_ops/).

