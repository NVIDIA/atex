"""Linear layers."""

from functools import partial
from typing import Optional, Tuple, Union, Iterable

from jax import lax
from jax import numpy as jnp
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import base_ops
from praxis.layers import flax_adapter

from fp8layers.flax import DenseGeneral

WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
JTensor = pytypes.JTensor


def generate_params_init(name: str, initializer: WeightInit):
   """Convert praxis init to flax-friendly init"""
   def kernel_init(key, shape, dtype):
     wp = WeightHParams(shape=shape, init=initializer, dtype=dtype)
     return base_layer.init_var(wp, key, name)
   return kernel_init

class Linear(base_layer.BaseLayer):
  """Linear Fp8 layer without bias.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
  """
  input_dims: int = 0
  output_dims: int = 0
  kernel_axes: Tuple[str, ...] = ()
  weight_init: Optional[WeightInit] = WeightInit.Xavier(scale=1.0)
  axis: Union[Iterable[int], int] = -1
  amax_history_length: int = 16
  logical_axes_rules: Tuple[Tuple, ...] = None

  def create_layer(self, name, flax_module_cls):
    """create_layer"""
    flax_module_p = pax_fiddle.Config(
        flax_adapter.FlaxModuleAdapter,
        module_factory_method=flax_module_cls,
        logical_axes_rules=self.logical_axes_rules,
        ici_mesh_shape=self.ici_mesh_shape,
        dcn_mesh_shape=self.dcn_mesh_shape,
        mesh_axis_names=self.mesh_axis_names)

    self.create_child(name, flax_module_p.clone())

  def setup(self) -> None:
    """setup"""
    super().setup()

    dense_general_cls = partial(
        DenseGeneral,
        features=self.output_dims,
        kernel_axes=self.kernel_axes,
        kernel_init=generate_params_init('kernel', self.weight_init),
        amax_history_length=self.amax_history_length,
        use_bias=False,
        bias_axes=None,
        axis=self.axis,
        dtype=self.dtype)

    self.create_layer("linear", dense_general_cls)

  def __call__(self, x: JTensor) -> JTensor:
    """__call__"""
    return self.linear(x)


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
    if self.has_bias:
      # Cublas fp8 matmul only supports bf16 bias for fp32 IO. For pattern
      # matching, we cast the bias to bf16 back and forth.
      if self.bias.dtype == jnp.float32:
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
