# NVIDIA/Atex: A TensorFlow Extension

This repository holds NVIDIA-maintained utilities to streamline training and
inference in TensorFlow. Some of the code here will be included in upstream
TensorFlow eventually. The intent of Atex is to make up-to-date utilities
available to users as quickly as possible.

## Contents

### 1. Fused Layer/Instance Normalization

`nv_norms.fused_layer_norm` and `nv_norms.fused_instance_norm` are fused
implementations designed to replace the `tf.keras.layers.LayerNormalization` and
`tfa.layers.InstanceNormalization`. The Keras and Addons implementations compose
many ops, such as Mul, Add, etc., to compute the stats and then scale/offset the
results. In contrast, the fused implementation provided in `nv_norms` computes
the norms in a single step, eliminating many expensive round-trips to device
memory and significantly improving performance. More in [nv_ops](./nv_ops/).

