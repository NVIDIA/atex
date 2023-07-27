"""Linear layers."""

from typing import Optional

from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import base_ops

from fp8layers.flax import fp8_dot

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
JTensor = pytypes.JTensor

class Linear(base_layer.BaseLayer):
  """Linear Fp8 layer without bias.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
  """
  input_dims: int = 0
  output_dims: int = 0
  amax_history_length: int = 16
  weight_init: Optional[WeightInit] = None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    self.create_variable(
        'w',
        WeightHParams(
            shape=[self.input_dims, self.output_dims],
            init=self.weight_init,
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt,
        ),
    )

    scale_args = {
        'shape': [1],
        'init': WeightInit.Constant(1.0),
        'dtype': jnp.float32,
        'mesh_shape': self.mesh_shape,
        'tensor_split_dims_mapping': None,
        'collections': ['fp8_params'],
    }
    amax_history_args = {
        'shape': [self.amax_history_length],
        'init': WeightInit.Constant(0.0),
        'dtype': jnp.float32,
        'mesh_shape': self.mesh_shape,
        'tensor_split_dims_mapping': None,
        'collections': ['fp8_params'],
    }
    self.create_variable(
        'input_amax_history', WeightHParams(**amax_history_args))
    self.create_variable(
        'kernel_amax_history', WeightHParams(**amax_history_args))
    self.create_variable(
        'output_grad_amax_history', WeightHParams(**amax_history_args))

    self.create_variable('input_scale', WeightHParams(**scale_args))
    self.create_variable('kernel_scale', WeightHParams(**scale_args))
    self.create_variable(
         'output_grad_scale', WeightHParams(**scale_args))

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    ap = self.activation_split_dims_mapping

    original_shape = inputs.shape
    assert len(original_shape) >= 2

    comp_dtype = self.fprop_dtype
    inputs = jnp.asarray(inputs, comp_dtype)
    kernel = jnp.asarray(self.theta.w, comp_dtype)

    # Reshape the inputs to 2D matrix to call the fp8_dot. The result will be
    # casted back at the end.
    inp = jnp.reshape(inputs, (-1, self.input_dims))

    out = fp8_dot(inp, kernel, comp_dtype,
                  self.theta.input_scale, self.theta.input_amax_history,
                  self.theta.kernel_scale, self.theta.kernel_amax_history,
                  self.theta.output_grad_scale,
                  self.theta.output_grad_amax_history)

    # Reshape back the outputs.
    out = jnp.reshape(out, (*original_shape[0:-1], self.output_dims))

    # Adjust sharding annotation during decoding.
    # TODO(pax): This logic should likely be lifted somewhere else.
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and out.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    out = base_layer.maybe_shard(out, ap_out, self.mesh_axis_names)
    return out

class Bias(base_layer.BaseLayer):
  """Bias layer.

  Attributes:
    dims: Depth of the input.
    bias_init: Init scale (constant) of bias terms.
  """
  dims: int = 0
  bias_init: Optional[float] = 0.0

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    self.create_variable(
        'b',
        WeightHParams(
            shape=[self.dims],
            init=WeightInit.Constant(self.bias_init),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt,
        ),
    )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Adds bias to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., dims].

    Returns:
      Inputs plus bias.
    """
    return inputs + self.theta.b


class FeedForward(base_layer.BaseLayer):
  """Feedforward layer with activation.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
    has_bias: Adds bias weights or not.
    linear_tpl: Linear layer params.
    activation_tpl: Activation layer params.
    bias_init: Init scale (constant) of bias terms.
    checkpoint_str: name to checkpoint the tensor output from nn.linear.
  """
  input_dims: int = 0
  output_dims: int = 0
  has_bias: bool = True
  linear_tpl: LayerTpl = template_field(Linear)
  bias_tpl: LayerTpl = template_field(Bias)
  activation_tpl: pax_fiddle.Config[
      activations.BaseActivation
  ] = template_field(activations.ReLU)
  weight_init: Optional[WeightInit] = None
  bias_init: Optional[float] = 0.0
  checkpoint_str: str | None = None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    ap = self.activation_split_dims_mapping
    linear_layer_p = self.linear_tpl.clone()
    linear_layer_p.set(
        input_dims=self.input_dims,
        output_dims=self.output_dims,
        weight_init=self.weight_init,
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone(),
    )
    # Provide type hint.
    self.linear: Linear
    self.create_child('linear', linear_layer_p)
    if self.has_bias:
      bias_layer_p = self.bias_tpl.clone()
      bias_layer_p.set(dims=self.output_dims, bias_init=self.bias_init)

      if self.mesh_shape is not None and ap.out is not None:
        wp_bias = [ap.out[-1]]
        bias_layer_p.weight_split_dims_mapping.wt = wp_bias
      # Provide type hint.
      self.bias: Bias
      self.create_child('bias', bias_layer_p)
    # Provide type hints
    self.activation: activations.BaseActivation
    self.create_child('activation', self.activation_tpl.clone())

  def __call__(self, inputs: JTensor) -> JTensor:
    projected_inputs = self.linear(inputs)
    if self.checkpoint_str is not None:
      projected_inputs = checkpoint_name(projected_inputs, self.checkpoint_str)
    if self.has_bias:
      # Cublas fp8 matmul only supports bf16 bias for fp32 IO. For pattern
      # matching, we cast the bias to bf16 back and forth. For now, we only do
      # this when the input shape is 2D, when there is no extra
      # reshape/transpose in-between the matmul and bias_add.
      if (len(inputs.shape) == 2 and self.bias.dtype == jnp.float32 and
          self.bias.fprop_dtype == jnp.float32):
        bias = self.bias.theta.b.astype(jnp.bfloat16)
        bias = bias.astype(jnp.float32)
        projected_inputs = projected_inputs + bias
      else:
        projected_inputs = self.bias(projected_inputs)
    output = self.activation(projected_inputs)
    return output

class MLPBlock(base_layer.BaseLayer):
  """Feedforward layer with activation.

  Attributes:
    num_layers: Number of FeedForward layers.
    hidden_dims: Dimension of hidden layers.
    ff_tpl: Feedforward layer params.
  """
  num_layers: int = 3
  hidden_dims: int = 128
  ff_tpl: LayerTpl = template_field(FeedForward)

  def setup(self) -> None:

    wp = self.weight_split_dims_mapping
    ap = self.activation_split_dims_mapping
    input_layer_p = self.ff_tpl.clone()
    input_layer_p.set(
        input_dims=self.ff_tpl.input_dims,
        output_dims=self.hidden_dims,
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone(),
    )
    hidden_layer_p = self.ff_tpl.clone()
    hidden_layer_p.set(
        input_dims=self.hidden_dims,
        output_dims=self.hidden_dims,
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone(),
    )
    output_layer_p = self.ff_tpl.clone()
    output_layer_p.set(
        input_dims=self.hidden_dims,
        output_dims=self.ff_tpl.output_dims,
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone(),
    )
    mlp_layers = [input_layer_p]
    for _ in range(self.num_layers - 2):
      mlp_layers.append(hidden_layer_p)
    mlp_layers.append(output_layer_p)
    self.create_children('mlp_layers', mlp_layers)

  def __call__(self, inputs: JTensor) -> JTensor:
    output = inputs
    for i in range(self.num_layers):
      output = self.mlp_layers[i](output)
    return output
