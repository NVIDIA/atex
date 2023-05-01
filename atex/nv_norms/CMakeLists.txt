# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 52 60 70 80 90)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(nv_norms LANGUAGES CUDA CXX)

list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter REQUIRED)


add_library(_fused_instance_norm_ops OBJECT
  cc/kernels/fused_instance_norm_op.cu
)
target_compile_definitions(_fused_instance_norm_ops PRIVATE GOOGLE_CUDA=1)
set_property(TARGET _fused_instance_norm_ops PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(_fused_layer_norm_ops OBJECT
  cc/kernels/fused_layer_norm_op.cu
)
target_compile_definitions(_fused_layer_norm_ops PRIVATE GOOGLE_CUDA=1)
set_property(TARGET _fused_layer_norm_ops PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(_fused_nv_norm_ops SHARED
  cc/ops/norm_ops.cc
  cc/kernels/fused_instance_norm_op.cc
  cc/kernels/fused_layer_norm_op.cc
  $<TARGET_OBJECTS:_fused_layer_norm_ops>
  $<TARGET_OBJECTS:_fused_instance_norm_ops>
)
set_target_properties(_fused_nv_norm_ops PROPERTIES PREFIX "")
target_compile_definitions(_fused_nv_norm_ops PRIVATE GOOGLE_CUDA=1)

# Compile flags (and include directories)
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE TF_COMPILE_FLAGS)
string(STRIP ${TF_COMPILE_FLAGS} TF_COMPILE_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TF_COMPILE_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TF_COMPILE_FLAGS} --expt-relaxed-constexpr")

# Link libraries
execute_process(COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(tf.__file__)" 
                OUTPUT_VARIABLE Tensorflow_LIB_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(Tensorflow_LIB_PATH ${Tensorflow_LIB_PATH} DIRECTORY)
list(APPEND TF_LINKER_LIBS "${Tensorflow_LIB_PATH}/libtensorflow_framework.so.2")
target_link_libraries(_fused_nv_norm_ops PRIVATE ${TF_LINKER_LIBS} CUDA::cudart)
