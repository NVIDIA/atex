/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
==============================================================================*/

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "fused_layer_norm_op.h"

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

namespace functor {

template <typename T, typename U>
struct FusedLayerNorm<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  U epsilon, int batch_dim, int feature_dim, Tensor* y_output,
                  Tensor* saved_mean_output, Tensor* saved_inv_var_output) {
    const auto N = x_input.dim_size(batch_dim);
    const auto D = x_input.dim_size(feature_dim);

    typename TTypes<T, 2>::ConstTensor x_in(x_input.tensor<T, 2>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec offset(offset_input.vec<U>());
    typename TTypes<T, 2>::Tensor y(y_output->tensor<T, 2>());
    typename TTypes<U>::Vec saved_mean(saved_mean_output->vec<U>());
    typename TTypes<U>::Vec saved_inv_var(saved_inv_var_output->vec<U>());

    const CPUDevice& d = context->eigen_device<CPUDevice>();

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> N_by_one(N, 1);
    Eigen::DSizes<Eigen::Index, 2> one_by_D(1, D);
    Eigen::array<int, 2> bcast_N({static_cast<int>(N), 1});
    Eigen::array<int, 2> bcast_D({1, static_cast<int>(D)});
    Eigen::array<int, 1> reduce_D({feature_dim});
#else
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1> > N_by_one;
    N_by_one.set(0, N);
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_D;
    one_by_D.set(1, D);
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1> > bcast_N;
    bcast_N.set(0, N);
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> bcast_D;
    bcast_D.set(1, D);
    Eigen::IndexList<Eigen::type2index<1> > reduce_D;
#endif

    Eigen::Tensor<float, 1, Eigen::RowMajor> mean(N);
    Eigen::Tensor<float, 1, Eigen::RowMajor> inv_var(N);

    U D_inv = static_cast<U>(1.0f / static_cast<U>(D));
    auto x = x_in.template cast<U>();
    mean.device(d) = x.sum(reduce_D) * D_inv;

    auto x_centered = x - mean.reshape(N_by_one).broadcast(bcast_D);

    inv_var.device(d) =
        (x_centered.square().sum(reduce_D) * D_inv + epsilon).rsqrt();

    auto scaling_factor = inv_var.reshape(N_by_one).broadcast(bcast_D) *
                          scale.reshape(one_by_D).broadcast(bcast_N);
    auto x_scaled = x_centered * scaling_factor;

    auto x_shifted = (x_scaled + offset.reshape(one_by_D).broadcast(bcast_N));

    y.device(d) = x_shifted.template cast<T>();
    saved_mean.device(d) = mean;
    saved_inv_var.device(d) = inv_var;
  }
};

template <typename T, typename U>
struct FusedLayerNormGrad<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& saved_mean_input,
                  const Tensor& saved_inv_var_input, U epsilon, int batch_dim,
                  int feature_dim, Tensor* x_backprop_output,
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

    typename TTypes<T, 2>::ConstTensor y_backprop(
        y_backprop_input.tensor<T, 2>());
    typename TTypes<T, 2>::ConstTensor x(x_input.tensor<T, 2>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec mean(saved_mean_input.vec<U>());
    typename TTypes<U>::ConstVec inv_var(saved_inv_var_input.vec<U>());
    typename TTypes<T, 2>::Tensor x_backprop(x_backprop_output->tensor<T, 2>());
    typename TTypes<U>::Vec offset_backprop(offset_backprop_output->vec<U>());

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    const int N = x.dimension(batch_dim);
    const int D = x.dimension(feature_dim);
    Eigen::DSizes<Eigen::Index, 2> N_by_D(N, D);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> one_by_D(1, D);
    Eigen::DSizes<Eigen::Index, 2> N_by_one(N, 1);
    Eigen::array<int, 2> bcast_N({N, 1});
    Eigen::array<int, 2> bcast_D({1, D});
    Eigen::array<int, 1> reduce_D({1});
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_D;
    one_by_D.set(1, D);
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1> > N_by_one;
    N_by_one.set(0, N);
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1> > bcast_N;
    bcast_N.set(0, N);
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> bcast_D;
    bcast_D.set(1, D);
    Eigen::IndexList<Eigen::type2index<1> > reduce_D;
#endif

    U D_inv = static_cast<U>(1.0f / static_cast<U>(D));
    auto y_backprop_N_by_D = y_backprop.template cast<U>();
    auto x_N_by_D = x.template cast<U>();

    // Eigen is notoriously bad at reducing outer dimension, so we materialize
    // all temporary tensors that require reduction, and then use Eigen redux
    // functor, that is optimized for this particular task.
    //
    // All reductions are of this type: [N, D] -> [D].
    using ScalarSum = Eigen::internal::scalar_sum_op<U>;
    const functor::ReduceOuterDimensions<T, U, U, ScalarSum> redux_sum_t;
    const functor::ReduceOuterDimensions<U, U, U, ScalarSum> redux_sum_u;

    auto scratch_dtype = DataTypeToEnum<U>::value;

    // Allocate a temporary workspace of [D] shape.
    Tensor scratch_one_by_D;
    OP_REQUIRES_OK(
        context, context->allocate_temp(scratch_dtype, {D}, &scratch_one_by_D));

    // Maybe allocate a temporary workspace of [N, D] shape.
    Tensor scratch_N_by_D;
    if (std::is_same<T, U>::value) {
      OP_REQUIRES(context, scratch_N_by_D.CopyFrom(*x_backprop_output, {N, D}),
                  errors::Internal("Failed to copy a tensor"));
    } else {
      OP_REQUIRES_OK(context, context->allocate_temp(scratch_dtype, {N, D},
                                                     &scratch_N_by_D));
    }

    typename TTypes<U, 2>::Tensor scratch_tensor(scratch_N_by_D.tensor<U, 2>());
    typename TTypes<U>::Vec scratch_vector(scratch_one_by_D.vec<U>());

    auto mean_N_by_D = mean.reshape(N_by_one).broadcast(bcast_D);
    auto x_centered = (x_N_by_D - mean_N_by_D);

    auto inv_var_N_by_D = inv_var.reshape(N_by_one).broadcast(bcast_D);
    auto x_scaled = x_centered * inv_var_N_by_D;

    // Compute `scale_backprop_output`:
    //   scale_backprop =
    //     (y_backprop_N_by_D * x_scaled).sum(reduce_dims)
    scratch_tensor.device(d) = y_backprop_N_by_D * x_scaled;
    redux_sum_u(d, N_by_D, scratch_N_by_D, scale_backprop_output);

    // Compute 'offset_backprop_output':
    //   offset_backprop =
    //     y_backprop_N_by_D.sum(reduce_dims)
    redux_sum_t(d, N_by_D, y_backprop_input, offset_backprop_output);

    // Note: the following formulas are used to compute the gradients for
    // x_backprop.
    //   x_backprop = dl_dx + dl_dvar * dvar_dx + dl_dmean * dmean_dx.

    // Compute 'dl_dx':
    //   dl_dx = dy * scale * ivar (shape = N, D)
    auto dl_dx = (y_backprop_N_by_D *
                  scale.reshape(one_by_D).broadcast(bcast_N) * inv_var_N_by_D)
                     .eval();

    // Compute 'dl_dvar':
    //   dl_dvar = reduce_D(dy * scale * x_centered * -0.5 * ivar^3) (shape = N,
    //   1)
    auto dl_dvar =
        ((dl_dx * x_centered * (-0.5f) * inv_var_N_by_D * inv_var_N_by_D)
             .sum(reduce_D))
            .eval();
    // Compute 'dvar_dx':
    //   dvar_dx = 2 * x_centered * D_inv (shape = N, D)
    auto dvar_dx = (2.f * x_centered * D_inv).eval();

    // Compute 'dl_mean':
    //   dl_mean = reduce_D(-1 * dy * scale * ivar) +
    //             reduce_D(dl_dvar * -2 / D * x_centered) (shape = N, 1)
    auto dl_dmean = (-1.f * dl_dx).sum(reduce_D).eval() +
                    (dl_dvar.reshape(N_by_one).broadcast(bcast_D) * (-2.f) *
                     D_inv * x_centered)
                        .sum(reduce_D)
                        .eval();
    U dmean_dx = 1.f * D_inv;

    auto dx = dl_dx + dl_dvar.reshape(N_by_one).broadcast(bcast_D) * dvar_dx +
              dl_dmean.reshape(N_by_one).broadcast(bcast_D) * dmean_dx;
    x_backprop.device(d) = dx.template cast<T>();
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define DECLARE_GPU_SPEC(T, U)                                              \
  template <>                                                               \
  void FusedLayerNorm<GPUDevice, T, U>::operator()(                         \
      OpKernelContext* context, const Tensor& x_input,                      \
      const Tensor& scale_input, const Tensor& offset_input, U epsilon,     \
      int batch_dim, int feature_dim, Tensor* y_output,                     \
      Tensor* saved_mean_output, Tensor* saved_inv_var_output);             \
  extern template struct FusedLayerNorm<GPUDevice, T, U>;                   \
  template <>                                                               \
  void FusedLayerNormGrad<GPUDevice, T, U>::operator()(                     \
      OpKernelContext* context, const Tensor& y_backprop_input,             \
      const Tensor& x_input, const Tensor& scale_input,                     \
      const Tensor& saved_mean_input, const Tensor& saved_inv_var_input,    \
      U epsilon, int batch_dim, int feature_dim, Tensor* x_backprop_output, \
      Tensor* scale_backprop_output, Tensor* offset_backprop_output);       \
  extern template struct FusedLayerNormGrad<GPUDevice, T, U>;

DECLARE_GPU_SPEC(float, float);
DECLARE_GPU_SPEC(Eigen::half, float);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace functor

namespace {
Status GetBatchAndFeatureSizes(const Tensor& x, const std::vector<int>& axis,
                               int64_t* num_batches, int64_t* num_features) {
  int rank = x.dims();
  std::vector<int> processed_axis;

  for (auto a : axis) {
    int positive_axis = a < 0 ? rank + a : a;
    if (positive_axis < 0 || positive_axis >= rank) {
      return errors::InvalidArgument("axis contains invalid value: ", a);
    }
    processed_axis.push_back(positive_axis);
  }
  std::sort(processed_axis.begin(), processed_axis.end());
  processed_axis.erase(
      std::unique(processed_axis.begin(), processed_axis.end()),
      processed_axis.end());
  if (processed_axis[0] != rank - processed_axis.size()) {
    return errors::InvalidArgument("axis is not packed from last dim.");
  }

  *num_features = 1;
  *num_batches = 1;
  for (int i = 0; i < rank; i++) {
    if (!processed_axis.empty() && i >= processed_axis[0]) {
      *num_features *= x.dim_size(i);
    } else {
      *num_batches *= x.dim_size(i);
    }
  }
  return Status::OK();
}
}  // namespace

template <typename Device, typename T, typename U>
class FusedLayerNormOp : public OpKernel {
 public:
  explicit FusedLayerNormOp(OpKernelConstruction* context) : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = U(epsilon);
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* context) override {
    Tensor x_input = context->input(0);
    Tensor scale_input = context->input(1);
    Tensor offset_input = context->input(2);

    OP_REQUIRES(context, x_input.dims() > 0,
                errors::InvalidArgument("input must be at least 1-dimensional",
                                        x_input.shape().DebugString()));

    int64_t num_batches;
    int64_t num_features;
    OP_REQUIRES_OK(context, GetBatchAndFeatureSizes(
                                x_input, axis_, &num_batches, &num_features));

    OP_REQUIRES(context, scale_input.NumElements() == num_features,
                errors::InvalidArgument(
                    "scale must have the same number of elements "
                    "as the features of x, got ",
                    scale_input.NumElements(), " and ", num_features));
    OP_REQUIRES(context, offset_input.NumElements() == num_features,
                errors::InvalidArgument(
                    "offset must have the same number of elements "
                    "as the features of x, got ",
                    offset_input.NumElements(), " and ", num_features));

    TensorShape original_input_shape = x_input.shape();
    OP_REQUIRES(
        context,
        x_input.CopyFrom(x_input, TensorShape({num_batches, num_features})),
        errors::InvalidArgument("Error during tensor copy."));
    OP_REQUIRES(context,
                scale_input.CopyFrom(scale_input, TensorShape({num_features})),
                errors::InvalidArgument("Error during tensor copy."));
    OP_REQUIRES(
        context,
        offset_input.CopyFrom(offset_input, TensorShape({num_features})),
        errors::InvalidArgument("Error during tensor copy."));

    const int batch_dim = 0;
    const int feature_dim = 1;

    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, x_input.shape(), &out));
    Tensor* saved_mean_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {num_batches}, &saved_mean_out));
    Tensor* saved_inv_var_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {num_batches},
                                                     &saved_inv_var_out));

    functor::FusedLayerNorm<Device, T, U>()(
        context, x_input, scale_input, offset_input, epsilon_, batch_dim,
        feature_dim, out, saved_mean_out, saved_inv_var_out);

    OP_REQUIRES(context, out->CopyFrom(*out, original_input_shape),
                errors::InvalidArgument("Error during tensor copy."));
  }

 private:
  U epsilon_;
  std::vector<int> axis_;
};

template <typename Device, typename T, typename U>
class FusedLayerNormGradOp : public OpKernel {
 public:
  explicit FusedLayerNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = U(epsilon);
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* context) override {
    Tensor y_backprop = context->input(0);
    Tensor x = context->input(1);
    Tensor scale = context->input(2);
    const Tensor& saved_mean = context->input(3);
    const Tensor& saved_inv_var = context->input(4);

    OP_REQUIRES(context, y_backprop.dims() > 0,
                errors::InvalidArgument("input must be at least 1-dimensional",
                                        y_backprop.shape().DebugString()));
    OP_REQUIRES(context, x.dims() > 0,
                errors::InvalidArgument("input must be at least 1-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(context, saved_mean.dims() == 1,
                errors::InvalidArgument("saved mean must be 1-dimensional",
                                        saved_mean.shape().DebugString()));
    OP_REQUIRES(
        context, saved_inv_var.dims() == 1,
        errors::InvalidArgument("saved inverted variance must be 1-dimensional",
                                saved_inv_var.shape().DebugString()));
    OP_REQUIRES(
        context, x.shape() == y_backprop.shape(),
        errors::InvalidArgument(
            "x and y_backprop must have same shape, but x has shape ",
            x.shape(), " and y_backprop has shape ", y_backprop.shape()));

    int64_t num_batches;
    int64_t num_features;
    OP_REQUIRES_OK(context, GetBatchAndFeatureSizes(x, axis_, &num_batches,
                                                    &num_features));

    OP_REQUIRES(
        context, scale.NumElements() == num_features,
        errors::InvalidArgument("scale must have the same number of elements "
                                "as the features of x, got ",
                                scale.NumElements(), " and ", num_features));
    OP_REQUIRES(context, saved_mean.NumElements() == num_batches,
                errors::InvalidArgument(
                    "reserve_space_1 must have the same number of "
                    "elements as the batches of x, got ",
                    saved_mean.NumElements(), " and ", num_batches));
    OP_REQUIRES(context, saved_inv_var.NumElements() == num_batches,
                errors::InvalidArgument(
                    "reserve_space_2 must have the same number of "
                    "elements as the channels of x, got ",
                    saved_inv_var.NumElements(), " and ", num_batches));

    TensorShape original_input_shape = x.shape();
    TensorShape original_scale_shape = scale.shape();
    OP_REQUIRES(context,
                x.CopyFrom(x, TensorShape({num_batches, num_features})),
                errors::InvalidArgument("Error during tensor copy."));
    OP_REQUIRES(context,
                y_backprop.CopyFrom(y_backprop,
                                    TensorShape({num_batches, num_features})),
                errors::InvalidArgument("Error during tensor copy."));
    OP_REQUIRES(context, scale.CopyFrom(scale, TensorShape({num_features})),
                errors::InvalidArgument("Error during tensor copy."));

    const int batch_dim = 0;
    const int feature_dim = 1;

    Tensor* x_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, x.shape(), &x_backprop));

    const TensorShape& flattened_scale_shape = scale.shape();
    Tensor* scale_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, flattened_scale_shape,
                                                     &scale_backprop));
    Tensor* offset_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, flattened_scale_shape,
                                                     &offset_backprop));

    functor::FusedLayerNormGrad<Device, T, U>()(
        context, y_backprop, x, scale, saved_mean, saved_inv_var, epsilon_,
        batch_dim, feature_dim, x_backprop, scale_backprop, offset_backprop);

    OP_REQUIRES(context,
                x_backprop->CopyFrom(*x_backprop, original_input_shape),
                errors::InvalidArgument("Error during tensor copy."));
    OP_REQUIRES(context,
                scale_backprop->CopyFrom(*scale_backprop, original_scale_shape),
                errors::InvalidArgument("Error during tensor copy."));
    OP_REQUIRES(
        context,
        offset_backprop->CopyFrom(*offset_backprop, original_scale_shape),
        errors::InvalidArgument("Error during tensor copy."));
  }

 private:
  U epsilon_;
  std::vector<int> axis_;
};

#define REGISTER_KERNELS(D, T, U)                             \
  REGISTER_KERNEL_BUILDER(Name("FusedLayerNorm")              \
                              .Device(DEVICE_##D)             \
                              .TypeConstraint<T>("T")         \
                              .TypeConstraint<U>("U"),        \
                          FusedLayerNormOp<D##Device, T, U>); \
  REGISTER_KERNEL_BUILDER(Name("FusedLayerNormGrad")          \
                              .Device(DEVICE_##D)             \
                              .TypeConstraint<T>("T")         \
                              .TypeConstraint<U>("U"),        \
                          FusedLayerNormGradOp<D##Device, T, U>);

REGISTER_KERNELS(CPU, float, float);
REGISTER_KERNELS(CPU, Eigen::half, float);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNELS(GPU, float, float);
REGISTER_KERNELS(GPU, Eigen::half, float);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

}  // namespace tensorflow
