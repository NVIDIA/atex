#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

# Setup the include path for building dynamic library.
if [[ -z "${PYVER}" ]]; then
  echo "Unknown python version: PYVER"; exit 1;
fi

# Build the wheel.
make nv_norms_pip_pkg

