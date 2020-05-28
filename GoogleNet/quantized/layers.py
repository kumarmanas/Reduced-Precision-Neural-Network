"""Each Quantized Layer requires a `input_quantizer` and `kernel_quantizer` that
describes the way of quantizing the activation of the previous layer and the weights
respectively.

If both `input_quantizer` and `kernel_quantizer` are `None` the layer
is equivalent to a full precision layer.
"""

import tensorflow as tf
from larq import utils
from larq.layers_base import (
    QuantizerBase,
)


@utils.register_keras_custom_object
class QuantDense(QuantizerBase, tf.keras.layers.Dense):
    """Just your regular densely-connected quantized NN layer.

    `QuantDense` implements the operation:
    `output = activation(dot(input_quantizer(input), kernel_quantizer(kernel)) + bias)`,
    where `activation` is the element-wise activation function passed as the
    `activation` argument, `kernel` is a weights matrix created by the layer, and `bias`
    is a bias vector created by the layer (only applicable if `use_bias` is `True`).
    `input_quantizer` and `kernel_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Dense`.

    !!! note ""
        If the input to the layer has a rank greater than 2, then it is flattened
        prior to the initial dot product with `kernel`.

    !!! example
        ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(
            QuantDense(
                32,
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
                input_shape=(16,),
            )
        )
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(
            QuantDense(
                32,
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
            )
        )
        ```

    # Arguments
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use. If you don't specify anything,
        no activation is applied (`a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    input_quantizer: Quantization function applied to the input of the layer.
    kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    metrics: An array of metrics to add to the layer. If `None` the metrics set in
        `larq.metrics.scope` are used.
        Currently only the `flip_ratio` metric is available.

    # Input shape
    N-D tensor with shape: `(batch_size, ..., input_dim)`. The most common situation
    would be a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
    N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D input with
    shape `(batch_size, input_dim)`, the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        metrics=None,
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            metrics=metrics,
            **kwargs,
        )

@utils.register_keras_custom_object
class QuantConv2D(QuantizerBase, tf.keras.layers.Conv2D):
    """2D quantized convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    `input_quantizer` and `kernel_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Conv2D`. If `use_bias` is True, a bias vector is created
    and added to the outputs. Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument
    `input_shape` (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures in
    `data_format="channels_last"`.

    # Arguments
    filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window. Can be a single integer
        to specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width. Can be a single integer to
        specify the same value for all spatial dimensions. Specifying any stride
        value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs. `channels_last` corresponds to
        inputs with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It defaults
        to the `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json`. If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate
        to use for dilated convolution. Can be a single integer to specify the same
        value for all spatial dimensions. Currently, specifying any `dilation_rate`
        value != 1 is incompatible with specifying any stride value != 1.
    activation: Activation function to use. If you don't specify anything,
        no activation is applied (`a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    input_quantizer: Quantization function applied to the input of the layer.
    kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to the kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.
    metrics: An array of metrics to add to the layer. If `None` the metrics set in
        `larq.metrics.scope` are used.
        Currently only the `flip_ratio` metric is available.

    # Input shape
    4D tensor with shape:
    `(samples, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
    4D tensor with shape:
    `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        metrics=None,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            metrics=metrics,
            **kwargs,
        )

