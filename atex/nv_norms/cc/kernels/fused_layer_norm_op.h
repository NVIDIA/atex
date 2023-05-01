/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORM_OP_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORM_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, typename U>
struct FusedLayerNorm {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  U epsilon, int batch_dim, int feature_dim, Tensor* y_output,
                  Tensor* saved_mean_output, Tensor* saved_inv_var_output) {}
};

template <typename Device, typename T, typename U>
struct FusedLayerNormGrad {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& saved_mean_input,
                  const Tensor& saved_inv_var_input, U epsilon, int batch_dim,
                  int feature_dim, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output,
                  Tensor* offset_backprop_output) {}
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORM_OP_H_
