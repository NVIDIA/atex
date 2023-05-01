/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
==============================================================================*/

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "fused_instance_norm_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/redux_functor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {
bool FormatFromSuccinctString(absl::string_view format_str,
                              TensorFormat* format) {
  int n = format_str.size();
  if (n > 2 && format_str[0] == 'N' &&
      (format_str[1] == 'C' || format_str[n - 1] == 'C')) {
    *format = format_str[1] == 'C' ? FORMAT_NCHW : FORMAT_NHWC;
    return true;
  }
  return false;
}
}  // end namespace

namespace functor {

template <typename T, typename U>
struct FusedInstanceNorm<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  U epsilon, const bool is_channel_first, Tensor* y_output,
                  Tensor* saved_mean_output, Tensor* saved_inv_var_output) {
    int batch_dim = 0;
    int channel_dim = is_channel_first ? 1 : 2;
    int feature_dim = is_channel_first ? 2 : 1;
    const auto N = x_input.dim_size(batch_dim);
    const auto C = x_input.dim_size(channel_dim);
    const auto D = x_input.dim_size(feature_dim);

    typename TTypes<T, 3>::ConstTensor x_in(x_input.tensor<T, 3>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec offset(offset_input.vec<U>());
    typename TTypes<T, 3>::Tensor y(y_output->tensor<T, 3>());
    typename TTypes<U, 2>::Tensor saved_mean(saved_mean_output->tensor<U, 2>());
    typename TTypes<U, 2>::Tensor saved_inv_var(
        saved_inv_var_output->tensor<U, 2>());

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    auto set_to_one = [=](const string mask) {
      Eigen::array<int64_t, 3> ret_vec({N, C, D});
      if (!is_channel_first) {
        std::swap(ret_vec[channel_dim], ret_vec[feature_dim]);
      }
      if (mask.find('N') != std::string::npos) {
        ret_vec[batch_dim] = 1;
      }
      if (mask.find('C') != std::string::npos) {
        ret_vec[channel_dim] = 1;
      }
      if (mask.find('D') != std::string::npos) {
        ret_vec[feature_dim] = 1;
      }
      return ret_vec;
    };

    Eigen::DSizes<Eigen::Index, 3> N_by_C_by_one(set_to_one("D"));
    Eigen::DSizes<Eigen::Index, 3> one_by_C_by_one(set_to_one("ND"));
    Eigen::array<int64_t, 3> bcast_D(set_to_one("NC"));
    Eigen::array<int64_t, 3> bcast_ND(set_to_one("C"));
    Eigen::array<int64_t, 1> reduce_D({feature_dim});

    Eigen::Tensor<float, 2, Eigen::RowMajor> mean(N, C);
    Eigen::Tensor<float, 2, Eigen::RowMajor> inv_var(N, C);

    U D_inv = static_cast<U>(1.0f / static_cast<U>(D));
    auto x = x_in.template cast<U>();
    mean.device(d) = x.sum(reduce_D) * D_inv;

    auto x_centered = x - mean.reshape(N_by_C_by_one).broadcast(bcast_D);

    inv_var.device(d) =
        (x_centered.square().sum(reduce_D) * D_inv + epsilon).rsqrt();

    auto scaling_factor = inv_var.reshape(N_by_C_by_one).broadcast(bcast_D) *
                          scale.reshape(one_by_C_by_one).broadcast(bcast_ND);
    auto x_scaled = x_centered * scaling_factor;

    auto x_shifted =
        (x_scaled + offset.reshape(one_by_C_by_one).broadcast(bcast_ND));

    y.device(d) = x_shifted.template cast<T>();
    saved_mean.device(d) = mean;
    saved_inv_var.device(d) = inv_var;
  }
};

template <typename T, typename U>
struct FusedInstanceNormGrad<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& saved_mean_input,
                  const Tensor& saved_inv_var_input, U epsilon,
                  const bool is_channel_first, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output,
                  Tensor* offset_backprop_output) {
    if (x_input.shape().num_elements() == 0) {
      auto out = scale_backprop_output->flat<U>();
      const CPUDevice& d = context->eigen_device<CPUDevice>();
      out.device(d) = out.constant(U(0));
      auto out1 = offset_backprop_output->flat<U>();
      out1.device(d) = out1.constant(U(0));
      return;
    }

    typename TTypes<T, 3>::ConstTensor y_backprop(
        y_backprop_input.tensor<T, 3>());
    typename TTypes<T, 3>::ConstTensor x(x_input.tensor<T, 3>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U, 2>::ConstTensor mean(saved_mean_input.tensor<U, 2>());
    typename TTypes<U, 2>::ConstTensor inv_var(
        saved_inv_var_input.tensor<U, 2>());
    typename TTypes<T, 3>::Tensor x_backprop(x_backprop_output->tensor<T, 3>());
    typename TTypes<U>::Vec offset_backprop(offset_backprop_output->vec<U>());

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    int batch_dim = 0;
    int channel_dim = is_channel_first ? 1 : 2;
    int feature_dim = is_channel_first ? 2 : 1;
    const auto N = x_input.dim_size(batch_dim);
    const auto C = x_input.dim_size(channel_dim);
    const auto D = x_input.dim_size(feature_dim);

    TensorShape input_tensor_shape =
        is_channel_first ? TensorShape({N, C, D})
                         : input_tensor_shape = TensorShape({N, D, C});

    auto set_to_one = [=](const string mask) {
      Eigen::array<int64_t, 3> ret_vec({N, C, D});
      if (!is_channel_first) {
        std::swap(ret_vec[channel_dim], ret_vec[feature_dim]);
      }
      if (mask.find('N') != std::string::npos) {
        ret_vec[batch_dim] = 1;
      }
      if (mask.find('C') != std::string::npos) {
        ret_vec[channel_dim] = 1;
      }
      if (mask.find('D') != std::string::npos) {
        ret_vec[feature_dim] = 1;
      }
      return ret_vec;
    };
    Eigen::DSizes<Eigen::Index, 3> N_by_C_by_D(set_to_one(""));
    Eigen::DSizes<Eigen::Index, 3> N_by_C_by_one(set_to_one("D"));
    Eigen::DSizes<Eigen::Index, 3> one_by_C_by_one(set_to_one("ND"));
    Eigen::array<int64_t, 3> bcast_D(set_to_one("NC"));
    Eigen::array<int64_t, 3> bcast_ND(set_to_one("C"));
    Eigen::array<int64_t, 1> reduce_D({feature_dim});

    U D_inv = static_cast<U>(1.0f / static_cast<U>(D));
    auto y_backprop_N_by_C_by_D = y_backprop.template cast<U>();
    auto x_N_by_C_by_D = x.template cast<U>();

    // Eigen is notoriously bad at reducing outer dimension, so we materialize
    // all temporary tensors that require reduction, and then use Eigen redux
    // functor, that is optimized for this particular task.
    //
    // All reductions are of this type: [N, C, D] -> [C].
    using ScalarSum = Eigen::internal::scalar_sum_op<U>;
    const functor::ReduceMiddleDimensions<T, U, U, ScalarSum,
                                          Eigen::internal::SumReducer<T>>
        redux_sum_mid_t;
    const functor::ReduceMiddleDimensions<U, U, U, ScalarSum,
                                          Eigen::internal::SumReducer<U>>
        redux_sum_mid_u;

    const functor::ReduceOuterDimensions<T, U, U, ScalarSum> redux_sum_out_t;
    const functor::ReduceOuterDimensions<U, U, U, ScalarSum> redux_sum_out_u;

    auto scratch_dtype = DataTypeToEnum<U>::value;

    Tensor scratch_one_by_C_by_one;
    OP_REQUIRES_OK(context, context->allocate_temp(scratch_dtype, {C},
                                                   &scratch_one_by_C_by_one));

    // Maybe allocate a temporary workspace of [N, C, D] shape.
    Tensor scratch_N_by_C_by_D;

    if (std::is_same<T, U>::value) {
      OP_REQUIRES(
          context,
          scratch_N_by_C_by_D.CopyFrom(*x_backprop_output, input_tensor_shape),
          errors::Internal("Failed to copy a tensor"));
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_temp(scratch_dtype, input_tensor_shape,
                                            &scratch_N_by_C_by_D));
    }

    typename TTypes<U, 3>::Tensor scratch_tensor(
        scratch_N_by_C_by_D.tensor<U, 3>());
    typename TTypes<U>::Vec scratch_vector(scratch_one_by_C_by_one.vec<U>());

    auto mean_N_by_C_by_D = mean.reshape(N_by_C_by_one).broadcast(bcast_D);
    auto x_centered = (x_N_by_C_by_D - mean_N_by_C_by_D);

    auto inv_var_N_by_C_by_D =
        inv_var.reshape(N_by_C_by_one).broadcast(bcast_D);
    auto x_scaled = x_centered * inv_var_N_by_C_by_D;

    // Compute `scale_backprop_output`:
    //   scale_backprop =
    //     (y_backprop_N_by_D * x_scaled).sum(reduce_dims)
    scratch_tensor.device(d) = y_backprop_N_by_C_by_D * x_scaled;
    if (is_channel_first) {
      redux_sum_mid_u(d, N_by_C_by_D, scratch_N_by_C_by_D,
                      scale_backprop_output, 1);
    } else {
      redux_sum_out_u(d, N_by_C_by_D, scratch_N_by_C_by_D,
                      scale_backprop_output);
    }

    // Compute 'offset_backprop_output':
    //   offset_backprop =
    //     y_backprop_N_by_C_by_D.sum(reduce_dims)
    if (is_channel_first) {
      redux_sum_mid_t(d, N_by_C_by_D, y_backprop_input, offset_backprop_output,
                      1);
    } else {
      redux_sum_out_t(d, N_by_C_by_D, y_backprop_input, offset_backprop_output);
    }

    // Note: the following formulas are used to compute the gradients for
    // x_backprop.
    //   x_backprop = dl_dx + dl_dvar * dvar_dx + dl_dmean * dmean_dx.

    // Compute 'dl_dx':
    //   dl_dx = dy * scale * ivar (shape = N, C, D)
    auto dl_dx = (y_backprop_N_by_C_by_D *
                  scale.reshape(one_by_C_by_one).broadcast(bcast_ND) *
                  inv_var_N_by_C_by_D)
                     .eval();

    // Compute 'dl_dvar':
    //   dl_dvar = reduce_D(dy * scale * x_centered * -0.5 * ivar^3) (shape = N,
    //   C, 1)
    auto dl_dvar = ((dl_dx * x_centered * (-0.5f) * inv_var_N_by_C_by_D *
                     inv_var_N_by_C_by_D)
                        .sum(reduce_D))
                       .eval();
    // Compute 'dvar_dx':
    //   dvar_dx = 2 * x_centered * D_inv (shape = N, D)
    auto dvar_dx = (2.f * x_centered * D_inv).eval();

    // Compute 'dl_mean':
    //   dl_mean = reduce_D(-1 * dy * scale * ivar) +
    //             reduce_D(dl_dvar * -2 / D * x_centered) (shape = N, 1)
    auto dl_dmean = (-1.f * dl_dx).sum(reduce_D).eval() +
                    (dl_dvar.reshape(N_by_C_by_one).broadcast(bcast_D) *
                     (-2.f) * D_inv * x_centered)
                        .sum(reduce_D)
                        .eval();
    U dmean_dx = D_inv;

    auto dx = dl_dx +
              dl_dvar.reshape(N_by_C_by_one).broadcast(bcast_D) * dvar_dx +
              dl_dmean.reshape(N_by_C_by_one).broadcast(bcast_D) * dmean_dx;
    x_backprop.device(d) = dx.template cast<T>();
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define DECLARE_GPU_SPEC(T, U)                                           \
  template <>                                                            \
  void FusedInstanceNorm<GPUDevice, T, U>::operator()(                   \
      OpKernelContext* context, const Tensor& x_input,                   \
      const Tensor& scale_input, const Tensor& offset_input, U epsilon,  \
      const bool is_channel_first, Tensor* y_output,                     \
      Tensor* saved_mean_output, Tensor* saved_inv_var_output);          \
  extern template struct FusedInstanceNorm<GPUDevice, T, U>;             \
  template <>                                                            \
  void FusedInstanceNormGrad<GPUDevice, T, U>::operator()(               \
      OpKernelContext* context, const Tensor& y_backprop_input,          \
      const Tensor& x_input, const Tensor& scale_input,                  \
      const Tensor& saved_mean_input, const Tensor& saved_inv_var_input, \
      U epsilon, const bool is_channel_first, Tensor* x_backprop_output, \
      Tensor* scale_backprop_output, Tensor* offset_backprop_output);    \
  extern template struct FusedInstanceNormGrad<GPUDevice, T, U>;

DECLARE_GPU_SPEC(float, float);
DECLARE_GPU_SPEC(Eigen::half, float);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace functor

template <typename Device, typename T, typename U>
class FusedInstanceNormOp : public OpKernel {
 public:
  explicit FusedInstanceNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromSuccinctString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = U(epsilon);
  }

  void Compute(OpKernelContext* context) override {
    Tensor x_input = context->input(0);
    const Tensor& scale_input = context->input(1);
    const Tensor& offset_input = context->input(2);

    OP_REQUIRES(context, x_input.dims() > 3,
                errors::InvalidArgument("input must be at least 4-dimensional",
                                        x_input.shape().DebugString()));
    OP_REQUIRES(context, scale_input.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale_input.shape().DebugString()));
    OP_REQUIRES(context, offset_input.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset_input.shape().DebugString()));

    const int batch_dim = 0;
    int channel_dim;
    int feature_leading_dim;
    int end_feature_dim;
    bool is_channel_first = false;
    // NCHW always has channel in 1st dimension.
    if (data_format_ == FORMAT_NCHW) {
      channel_dim = 1;
      feature_leading_dim = channel_dim + 1;
      end_feature_dim = x_input.dims();
      is_channel_first = true;
    } else {
      channel_dim = x_input.dims() - 1;
      feature_leading_dim = batch_dim + 1;
      end_feature_dim = channel_dim;
    }
    int64_t num_channels = x_input.dim_size(channel_dim);
    int64_t num_batches = x_input.dim_size(batch_dim);
    int64_t num_features = 1;

    for (int i = feature_leading_dim; i < end_feature_dim; ++i) {
      num_features *= x_input.dim_size(i);
    }

    TensorShape original_input_shape = x_input.shape();
    TensorShape input_copy;
    if (data_format_ == FORMAT_NCHW) {
      input_copy = TensorShape({num_batches, num_channels, num_features});
    } else {
      input_copy = TensorShape({num_batches, num_features, num_channels});
    }

    OP_REQUIRES(context, x_input.CopyFrom(x_input, input_copy),
                errors::InvalidArgument("Error during tensor copy."));

    OP_REQUIRES(context, scale_input.NumElements() == num_channels,
                errors::InvalidArgument(
                    "scale must have the same number of elements "
                    "as the channels of x, got ",
                    scale_input.NumElements(), " and ", num_channels));
    OP_REQUIRES(context, offset_input.NumElements() == num_channels,
                errors::InvalidArgument(
                    "offset must have the same number of elements "
                    "as the channels of x, got ",
                    offset_input.NumElements(), " and ", num_channels));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, x_input.shape(), &out));
    Tensor* saved_mean_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {num_batches, num_channels},
                                            &saved_mean_out));
    Tensor* saved_inv_var_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, {num_batches, num_channels},
                                            &saved_inv_var_out));
    functor::FusedInstanceNorm<Device, T, U>()(
        context, x_input, scale_input, offset_input, epsilon_, is_channel_first,
        out, saved_mean_out, saved_inv_var_out);

    OP_REQUIRES(context, out->CopyFrom(*out, original_input_shape),
                errors::InvalidArgument("Error during tensor copy."));
  }

 private:
  TensorFormat data_format_;
  U epsilon_;
};

template <typename Device, typename T, typename U>
class FusedInstanceNormGradOp : public OpKernel {
 public:
  explicit FusedInstanceNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromSuccinctString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = U(epsilon);
  }

  void Compute(OpKernelContext* context) override {
    Tensor y_backprop = context->input(0);
    Tensor x = context->input(1);
    const Tensor& scale = context->input(2);
    const Tensor& saved_mean = context->input(3);
    const Tensor& saved_inv_var = context->input(4);

    OP_REQUIRES(context, y_backprop.dims() > 3,
                errors::InvalidArgument("input must be at least 4-dimensional",
                                        y_backprop.shape().DebugString()));
    OP_REQUIRES(context, x.dims() > 1,
                errors::InvalidArgument("input must be at least 2-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(context, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(context, saved_mean.dims() == 2,
                errors::InvalidArgument("saved mean must be 2-dimensional",
                                        saved_mean.shape().DebugString()));
    OP_REQUIRES(
        context, saved_inv_var.dims() == 2,
        errors::InvalidArgument("saved inverted variance must be 2-dimensional",
                                saved_inv_var.shape().DebugString()));
    OP_REQUIRES(
        context, x.shape() == y_backprop.shape(),
        errors::InvalidArgument(
            "x and y_backprop must have same shape, but x has shape ",
            x.shape(), " and y_backprop has shape ", y_backprop.shape()));

    // For the input tensor, we treat the the 0th dimension as the batch and 
    // channel being the last or second dimension. The rest are treated as 
    // spatial dimensions(e.g. height, width and depth etc.).

    const int batch_dim = 0;
    int channel_dim;
    int feature_leading_dim;
    int end_feature_dim;
  
    if (data_format_ == FORMAT_NCHW) {
      channel_dim = 1;
      feature_leading_dim = channel_dim + 1;
      end_feature_dim = x.shape().dims();
    } else {
      channel_dim = x.shape().dims() - 1;
      feature_leading_dim = batch_dim + 1;
      end_feature_dim = channel_dim;
    }
    int64_t num_channels = x.dim_size(channel_dim);
    int64_t num_batches = x.dim_size(batch_dim);
    int64_t num_features = 1;
    for (int i = feature_leading_dim; i < end_feature_dim; ++i) {
      num_features *= x.dim_size(i);
    }
    TensorShape original_input_shape = x.shape();
    TensorShape input_copy;
    if (data_format_ == FORMAT_NCHW) {
      input_copy = TensorShape({num_batches, num_channels, num_features});
    } else {
      input_copy = TensorShape({num_batches, num_features, num_channels});
    }

    OP_REQUIRES(context, x.CopyFrom(x, input_copy),
                errors::InvalidArgument("Error during tensor copy."));
    OP_REQUIRES(context, y_backprop.CopyFrom(y_backprop, input_copy),
                errors::InvalidArgument("Error during tensor copy."));

    OP_REQUIRES(
        context, scale.NumElements() == num_channels,
        errors::InvalidArgument("scale must have the same number of elements "
                                "as the channels of x, got ",
                                scale.NumElements(), " and ", num_channels));
    OP_REQUIRES(
        context, saved_mean.NumElements() == num_batches * num_channels,
        errors::InvalidArgument("reserve_space_1 must have the same number of "
                                "elements as the batches*channels of x, got ",
                                saved_mean.NumElements(), " and ",
                                num_batches * num_channels));
    OP_REQUIRES(
        context, saved_inv_var.NumElements() == num_batches * num_channels,
        errors::InvalidArgument("reserve_space_2 must have the same number of "
                                "elements as the batches * channels of x, got ",
                                saved_inv_var.NumElements(), " and ",
                                num_batches * num_channels));

    Tensor* x_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, x.shape(), &x_backprop));

    const TensorShape& scale_offset_shape = scale.shape();
    Tensor* scale_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, scale_offset_shape,
                                                     &scale_backprop));
    Tensor* offset_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, scale_offset_shape,
                                                     &offset_backprop));

    functor::FusedInstanceNormGrad<Device, T, U>()(
        context, y_backprop, x, scale, saved_mean, saved_inv_var, epsilon_,
        data_format_ == FORMAT_NCHW, x_backprop, scale_backprop,
        offset_backprop);

    OP_REQUIRES(context,
                x_backprop->CopyFrom(*x_backprop, original_input_shape),
                errors::InvalidArgument("Error during tensor copy."));
  }

 private:
  TensorFormat data_format_;
  U epsilon_;
};

#define REGISTER_KERNELS(D, T, U)                                \
  REGISTER_KERNEL_BUILDER(Name("FusedInstanceNorm")              \
                              .Device(DEVICE_##D)                \
                              .TypeConstraint<T>("T")            \
                              .TypeConstraint<U>("U"),           \
                          FusedInstanceNormOp<D##Device, T, U>); \
  REGISTER_KERNEL_BUILDER(Name("FusedInstanceNormGrad")          \
                              .Device(DEVICE_##D)                \
                              .TypeConstraint<T>("T")            \
                              .TypeConstraint<U>("U"),           \
                          FusedInstanceNormGradOp<D##Device, T, U>);

REGISTER_KERNELS(CPU, float, float);
REGISTER_KERNELS(CPU, Eigen::half, float);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNELS(GPU, float, float);
REGISTER_KERNELS(GPU, Eigen::half, float);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

}  // namespace tensorflow
