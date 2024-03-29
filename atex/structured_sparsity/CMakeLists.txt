# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(structured_sparsity LANGUAGES CUDA CXX)

list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(
  structured_sparsity SHARED
  kernels/structured_sparsity.cu
)
target_link_libraries(structured_sparsity PUBLIC CUDA::cudart)
