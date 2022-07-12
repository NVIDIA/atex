#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

# Setup the include path for building dynamic library.
mkdir -p /usr/local/lib/python3.8/dist-packages/tensorflow/include/third_party/gpus/cuda

ln -s /usr/local/cuda/include /usr/local/lib/python3.8/dist-packages/tensorflow/include/third_party/gpus/cuda

# Build the wheel.
make nv_norms_pip_pkg

# Run the python test.
make layer_norm_test
make instance_norm_test
