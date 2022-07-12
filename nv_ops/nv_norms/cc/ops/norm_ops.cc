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
Status FusedLayerNormShape(shape_inference::InferenceContext* c) {
  ShapeHandle x_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &x_shape));

  // For the input tensor, the first dim is the batch and the remaining dims are
  // the features. There is no batch dim for 1D tensors.
  const int x_rank = c->Rank(x_shape);
  int num_batches = 1;
  if (x_rank > 1) {
    num_batches = c->Value(c->Dim(x_shape, 0));
  }
  const DimensionHandle batch_dim = c->MakeDim(num_batches);

  c->set_output(0, x_shape);
  c->set_output(1, c->Vector(batch_dim));
  c->set_output(2, c->Vector(batch_dim));
  return Status::OK();
}

Status FusedLayerNormGradShape(shape_inference::InferenceContext* c) {
  ShapeHandle x_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &x_shape));

  // For the input tensor, the first dim is the batch and the remaining dims are
  // the features. There is no batch dim for 1D tensors.
  const int x_rank = c->Rank(x_shape);
  int num_features = c->Value(c->Dim(x_shape, x_rank - 1));
  if (x_rank > 1) {
    for (int i = x_rank - 2; i >= 1; i--) {
      num_features *= c->Value(c->Dim(x_shape, i));
    }
  }
  const DimensionHandle channel_dim = c->MakeDim(num_features);

  c->set_output(0, x_shape);
  c->set_output(1, c->Vector(channel_dim));
  c->set_output(2, c->Vector(channel_dim));
  return Status::OK();
}

Status FusedInstanceNormShape(shape_inference::InferenceContext* c) {
  // For the input tensor, the first dim is the batch and the remaining dims are
  // the features. Always assume at least 4D tensor, NHWC or NCHW.
  ShapeHandle x_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 4, &x_shape));

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  const int x_rank = c->Rank(x_shape);
  int channel_dim_index = s.ok() && data_format == "NC.." ? 1 : x_rank - 1;

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
  int channel_dim_index = s.ok() && data_format == "NC.." ? 1 : x_rank - 1;

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
    .SetShapeFn(shape_inference::FusedLayerNormGradShape);

REGISTER_OP("FusedInstanceNorm")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Output("y: T")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Attr("T: {half, float}")
    .Attr(GetConvnetDataFormat2D3DAttrString())
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
    .Attr(GetConvnetDataFormat2D3DAttrString())
    .Attr("T: {half, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.001")
    .SetShapeFn(shape_inference::FusedInstanceNormGradShape);
}  // namespace tensorflow
