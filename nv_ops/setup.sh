#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

# Setup the include path for building dynamic library.
if [[ -z "${PYVER}" ]]; then
  echo "Unknown python version: PYVER"; exit 1;
fi

mkdir -p /usr/local/lib/python${PYVER}/dist-packages/tensorflow/include/third_party/gpus/cuda

ln -s /usr/local/cuda/include /usr/local/lib/python${PYVER}/dist-packages/tensorflow/include/third_party/gpus/cuda

# Build the wheel.
make nv_norms_pip_pkg

