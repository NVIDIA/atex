/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
==============================================================================*/

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/redux_functor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

namespace shape_inference {

namespace {
Status GetBatchAndFeatureSizes(shape_inference::InferenceContext* c,
                               const ShapeHandle& x_shape, int* num_batches,
                               int* num_features) {
  std::vector<int> axis;
  TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));

  const int rank = c->Rank(x_shape);
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
      *num_features *= c->Value(c->Dim(x_shape, i));
    } else {
      *num_batches *= c->Value(c->Dim(x_shape, i));
    }
  }
  return Status::OK();
}
}  // namespace

Status FusedLayerNormShape(shape_inference::InferenceContext* c) {
  ShapeHandle x_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &x_shape));

  int num_batches;
  int num_features;
  TF_RETURN_IF_ERROR(
      GetBatchAndFeatureSizes(c, x_shape, &num_batches, &num_features));
  const DimensionHandle batch_dim = c->MakeDim(num_batches);

  c->set_output(0, x_shape);
  c->set_output(1, c->Vector(batch_dim));
  c->set_output(2, c->Vector(batch_dim));
  return Status::OK();
}

Status FusedLayerNormGradShape(shape_inference::InferenceContext* c) {
  ShapeHandle x_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &x_shape));

  int num_batches;
  int num_features;
  TF_RETURN_IF_ERROR(
      GetBatchAndFeatureSizes(c, x_shape, &num_batches, &num_features));

  const DimensionHandle channel_dim = c->MakeDim(num_features);

  c->set_output(0, x_shape);
  c->set_output(1, c->Vector(channel_dim));
  c->set_output(2, c->Vector(channel_dim));
  return Status::OK();
}

Status FusedInstanceNormShape(shape_inference::InferenceContext* c) {
  // For the input tensor, the first dim is the batch and the remaining dims are
  // the features. Always assume at least 4D tensor.
  ShapeHandle x_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 4, &x_shape));

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  const int x_rank = c->Rank(x_shape);
  int channel_dim_index;
  if (s.ok() && data_format.size() > 2 && data_format[0] == 'N' &&
      (data_format[1] == 'C' || data_format[data_format.size() - 1] == 'C')) {
    channel_dim_index = data_format[1] == 'C' ? 1 : x_rank - 1;
  } else {
    return errors::InvalidArgument(
        "We only accept 'NC...' or 'N...C' data format, but got",
        (s.ok() ? data_format : ""));
  }

  int num_batches = c->Value(c->Dim(x_shape, 0));

  const DimensionHandle batch_dim = c->MakeDim(num_batches);
  const DimensionHandle channel_dim = c->Dim(x_shape, channel_dim_index);

  auto mean_var_shape = c->MakeShape({batch_dim, channel_dim});
  c->set_output(0, x_shape);
  c->set_output(1, mean_var_shape);
  c->set_output(2, mean_var_shape);
  return Status::OK();
}

Status FusedInstanceNormGradShape(shape_inference::InferenceContext* c) {
  ShapeHandle x_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 4, &x_shape));

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  const int x_rank = c->Rank(x_shape);
  int channel_dim_index;
  if (s.ok() && data_format.size() > 2 && data_format[0] == 'N' &&
      (data_format[1] == 'C' || data_format[data_format.size() - 1] == 'C')) {
    channel_dim_index = data_format[1] == 'C' ? 1 : x_rank - 1;
  } else {
    return errors::InvalidArgument(
        "We only accept 'NC...' or 'N...C' data format, but got",
        (s.ok() ? data_format : ""));
  }

  DimensionHandle channel_dim = c->Dim(x_shape, channel_dim_index);

  c->set_output(0, x_shape);
  c->set_output(1, c->Vector(channel_dim));
  c->set_output(2, c->Vector(channel_dim));
  return Status::OK();
}

}  // namespace shape_inference

REGISTER_OP("FusedLayerNorm")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Output("y: T")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Attr("T: {half, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.001")
    .Attr("axis: list(int) = [-1]")
    .SetShapeFn(shape_inference::FusedLayerNormShape);

REGISTER_OP("FusedLayerNormGrad")
    .Input("y_backprop: T")
    .Input("x: T")
    .Input("scale: U")
    .Input("reserve_space_1: U")
    .Input("reserve_space_2: U")
    .Output("x_backprop: T")
    .Output("scale_backprop: U")
    .Output("offset_backprop: U")
    .Attr("T: {half, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.001")
    .Attr("axis: list(int) = [-1]")
    .SetShapeFn(shape_inference::FusedLayerNormGradShape);

REGISTER_OP("FusedInstanceNorm")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Output("y: T")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Attr("T: {half, float}")
    .Attr("data_format: string = 'N...C' ")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.001")
    .SetShapeFn(shape_inference::FusedInstanceNormShape);

REGISTER_OP("FusedInstanceNormGrad")
    .Input("y_backprop: T")
    .Input("x: T")
    .Input("scale: U")
    .Input("reserve_space_1: U")
    .Input("reserve_space_2: U")
    .Output("x_backprop: T")
    .Output("scale_backprop: U")
    .Output("offset_backprop: U")
    .Attr("data_format: string = 'N...C' ")
    .Attr("T: {half, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.001")
    .SetShapeFn(shape_inference::FusedInstanceNormGradShape);
}  // namespace tensorflow
