"""Contains the custom dense layer for XLA fp8."""

import tensorflow as tf

from tensorflow.python.framework import dtypes

def get_fp8_max(fp8_dtype):
  assert fp8_dtype in (dtypes.float8_e4m3fn, dtypes.float8_e5m2)
  if fp8_dtype == dtypes.float8_e4m3fn:
    return dtypes.float8_e4m3fn.max
  return dtypes.float8_e5m2.max

def quantize(x, q_dtype, scale, compute_dtype):
  dtype_max = get_fp8_max(q_dtype)

  # The x might be some variable whose real dtype is determined by the global
  # policy, e.g., the mixed precision. So, we don't use x.dtype but pass in the
  # compute_dtype from the policy.
  scaled_x = x / tf.cast(scale, compute_dtype)
  clipped_x = tf.clip_by_value(scaled_x, -dtype_max, dtype_max)

  return tf.cast(clipped_x, q_dtype)

def dequantize(x, dq_dtype, scale):
  return tf.cast(x, dq_dtype) * tf.cast(scale, dq_dtype)

def quantize_dequantize(x, q_dtype, scale, compute_dtype):
  qx = quantize(x, q_dtype, scale, compute_dtype)
  return dequantize(qx, compute_dtype, scale)

def compute_scale(amax, scale, fp8_max, margin=0):
  """Default function to convert amax to scaling factor."""
  exp = tf.math.floor(tf.experimental.numpy.log2(fp8_max / amax)) - margin
  sf = tf.math.round(tf.math.pow(2., tf.math.abs(exp)))
  sf = tf.where(amax > 0.0, sf, scale)
  sf = tf.where(tf.math.is_finite(amax), sf, scale)
  sf = tf.where(exp < 0, 1.0 / sf, sf)
  # The scaling factor we need equals to the notion of "scale_inv" in
  # TransformerEngine. So, we convert the sf to its reciprocal.
  return 1.0 / sf

def update_scale_and_amax_history(x, q_dtype, scale_var, amax_history_var):
  dtype_max = get_fp8_max(q_dtype)

  amax_update = tf.cast(tf.math.reduce_max(tf.math.abs(x)), scale_var.dtype)
  amax_history_update = tf.tensor_scatter_nd_update(
      tf.roll(amax_history_var, shift=-1, axis=0), [[0]], [amax_update])
  amax_history_var.assign(amax_history_update)

  amax_from_history = tf.reduce_max(amax_history_var, axis=0)
  scale_update = compute_scale(amax_from_history, scale_var.value(), dtype_max)
  scale_var.assign(scale_update)

def qdq_and_update(x, q_dtype, scale_var, amax_history_var, compute_dtype):
  qdq_x = quantize_dequantize(x, q_dtype, scale_var.value(), compute_dtype)
  update_scale_and_amax_history(x, q_dtype, scale_var, amax_history_var)
  return qdq_x

class Dense(tf.keras.layers.Dense):
    """
    Applies a linear transformation to the incoming data :math:`y = xW + b`

    On NVIDIA GPUs it is a drop-in replacement for `tf.keras.layers.Dense` in
    the jitting functions to take advantage of XLA fp8.
    
    Example:
    ```python
      dense = Dense(32)
      @tf.function(jit_compile=True)
      def _infer_step(x):
        return dense(x)
      dense(x) # Will use fp8 matmul on NVIDIA Hopper GPUs.
    ```

    Args:
      units: Positive integer, dimensionality of the output space.
      amax_history_length: Positive integer, The length of the amax history
        window used for scaling factor computation.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        amax_history_length=16,
        **kwargs,
    ):
      super().__init__(units, **kwargs)
      self.amax_history_length = amax_history_length

    def build(self, input_shape):
      super().build(input_shape)

      amax_history_kwargs = {
          "shape": (self.amax_history_length,),
          "initializer": "zeros",
          "trainable": False,
          "experimental_autocast": False,
      }
      scale_kwargs = {
          "shape": (),
          "initializer": "ones",
          "trainable": False,
          "experimental_autocast": False,
      }
      self.input_amax_history = self.add_weight(
          "input_amax_history", **amax_history_kwargs)
      self.input_scale = self.add_weight("input_scale", **scale_kwargs)

      self.kernel_amax_history = self.add_weight(
          "kernel_amax_history", **amax_history_kwargs)
      self.kernel_scale = self.add_weight("kernel_scale", **scale_kwargs)

      self.output_grad_amax_history = self.add_weight(
          "output_grad_amax_history", **amax_history_kwargs)
      self.output_grad_scale = self.add_weight(
          "output_grad_scale", **scale_kwargs)

      self.built = True

    @tf.custom_gradient
    def in_qdq(self, inp):
      """Quantize-dequantize the input but not its gradient."""
      qin = qdq_and_update(inp, dtypes.float8_e4m3fn, self.input_scale,
                           self.input_amax_history, self._compute_dtype_object)

      def grad(in_grad):
        return in_grad

      return qin, grad

    @tf.custom_gradient
    def out_qdq(self, out):
      """Quantize-dequantize the output gradient but not the output."""

      def grad(out_grad):
        return qdq_and_update(out_grad, dtypes.float8_e5m2,
                              self.output_grad_scale,
                              self.output_grad_amax_history,
                              self._compute_dtype_object)
      return out, grad

    @tf.custom_gradient
    def kernel_qdq(self, kernel):
      """Quantize-dequantize the kernel but not its gradient."""

      qkernel = qdq_and_update(kernel, dtypes.float8_e4m3fn, self.kernel_scale,
                               self.kernel_amax_history,
                               self._compute_dtype_object)

      def grad(kernel_grad, variables=None):
        return kernel_grad

      return qkernel, grad

    def call(self, inputs):
      if (isinstance(inputs, tf.RaggedTensor) or
          isinstance(inputs, tf.SparseTensor)):
        return super().call(inputs)
        
      if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
        inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
      else:
        # Under non-mixed precision cases, F32 bias has to be converted to BF16
        # first to get the biasAdd fusion support. ref. PR
        # https://github.com/tensorflow/tensorflow/pull/60306 
        bias_bf16 = tf.cast(self.bias, dtypes.bfloat16)
        self.bias = tf.cast(bias_bf16, dtypes.float32)

      # We explicitly reshape the inputs to 2D before the qdq to avoid the
      # complex cases where the reshape/transpose are inserted in between the
      # qdq and matmul. Similarly, we apply the reshape after the qdq of output.
      original_shape = inputs.shape
      assert original_shape.rank >= 2
      a = tf.reshape(inputs, shape=(-1, original_shape[-1]))

      outputs = tf.matmul(a=self.in_qdq(a),
                          b=self.kernel_qdq(self.kernel))
      // out_qdq is placed immediately after matmul for the sake of pattern 
      // matching in gemm_rewrite. That way, the qdq is adjacent to the matmul_bprop
      // in back prop.
      outputs = self.out_qdq(outputs)
      
      if self.use_bias:
        outputs = tf.nn.bias_add(outputs, self.bias)

      if self.activation is not None:
        outputs = self.activation(outputs)

      outputs = tf.reshape(outputs,
                           shape=(*original_shape[0:-1], outputs.shape[-1]))

      return outputs

