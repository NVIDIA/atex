# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ==============================================================================

CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
DEFAULT_COMPUTE_CAPABILITIES = 52,60,70,80,90

COMMON_SRCS = $(filter-out nv_norms/cc/kernels/norm_util.h,$(wildcard nv_norms/cc/kernels/*.h)) $(wildcard nv_norms/cc/ops/*.cc)
LN_SRCS = nv_norms/cc/kernels/fused_layer_norm_op.cc ${COMMON_SRCS}
IN_SRCS = nv_norms/cc/kernels/fused_instance_norm_op.cc ${COMMON_SRCS}

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++17
NVCCFLAGS = -std=c++17 -c

comma:=,
ATEX_COMPUTE_CAPABILITIES ?= $(DEFAULT_COMPUTE_CAPABILITIES)
ATEX_COMPUTE_CAPABILITIES := $(subst $(comma), ,$(ATEX_COMPUTE_CAPABILITIES))
$(foreach SM,$(ATEX_COMPUTE_CAPABILITIES),\
    $(eval NVCCFLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)))
MAX_SM = $(lastword $(sort $(ATEX_COMPUTE_CAPABILITIES)))
ifneq ($(MAX_SM),)
NVCCFLAGS += -gencode arch=compute_$(MAX_SM),code=compute_$(MAX_SM)
endif

LDFLAGS = -shared ${TF_LFLAGS}

LN_GPU_ONLY_TARGET_LIB = nv_norms/python/ops/_fused_layer_norm_ops.cu.o
IN_GPU_ONLY_TARGET_LIB = nv_norms/python/ops/_fused_instance_norm_ops.cu.o

TARGET_LIB = nv_norms/python/ops/_fused_nv_norm_ops.so

$(LN_GPU_ONLY_TARGET_LIB): nv_norms/cc/kernels/fused_layer_norm_op.cu.cc
	$(NVCC) $(NVCCFLAGS) -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

$(IN_GPU_ONLY_TARGET_LIB): nv_norms/cc/kernels/fused_instance_norm_op.cu.cc
	$(NVCC) $(NVCCFLAGS) -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

$(TARGET_LIB): $(LN_SRCS) $(IN_SRCS) $(LN_GPU_ONLY_TARGET_LIB) $(IN_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

layer_norm_test: tests/fused_layer_norm_test.py nv_norms/python/ops/nv_norm_ops.py $(TARGET_LIB)
	$(PYTHON_BIN_PATH) tests/fused_layer_norm_test.py

instance_norm_test: tests/fused_instance_norm_test.py nv_norms/python/ops/nv_norm_ops.py $(TARGET_LIB)
	$(PYTHON_BIN_PATH) tests/fused_instance_norm_test.py

nv_norms_op: $(TARGET_LIB)

nv_norms_pip_pkg: $(TARGET_LIB)
	./build_pip_pkg.sh make artifacts

clean:
	rm -f $(LN_GPU_ONLY_TARGET_LIB) $(IN_GPU_ONLY_TARGET_LIB) $(TARGET_LIB) artifacts/*.whl
