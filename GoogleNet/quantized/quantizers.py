"""A Quantizer defines the way of transforming a full precision input to a
quantized output and the pseudo-gradient method used for the backwards pass.

Quantizers can either be used through quantizer arguments that are supported
for Larq layers, such as `input_quantizer` and `kernel_quantizer`; or they
can be used similar to activations, i.e. either through an `Activation` layer,
or through the `activation` argument supported by all forward layer:

```python
import tensorflow as tf
import larq as lq
...
x = lq.layers.QuantDense(64, activation=None)(x)
x = lq.layers.QuantDense(64, input_quantizer="ste_sign")(x)
```

is equivalent to:

```python
x = lq.layers.QuantDense(64)(x)
x = tf.keras.layers.Activation("ste_sign")(x)
x = lq.layers.QuantDense(64)(x)
```

as well as:

```python
x = lq.layers.QuantDense(64, activation="ste_sign")(x)
x = lq.layers.QuantDense(64)(x)
```

We highly recommend using the first of these formulations: for the
other two formulations, intermediate layers - like batch normalization or
average pooling - and shortcut connections may result in non-binary input
to the convolutions.

Quantizers can either be referenced by string or called directly.
The following usages are equivalent:

```python
lq.layers.QuantDense(64, kernel_quantizer="ste_sign")
```
```python
lq.layers.QuantDense(64, kernel_quantizer=lq.quantizers.ste_sign)
```
```python
lq.layers.QuantDense(64, kernel_quantizer=lq.quantizers.SteSign(clip_value=1.0))
```
"""

import tensorflow as tf
from larq import utils, math

try:
    from tensorflow import is_tensor
except:  # pragma: no cover
    # For compatibility with TensorFlow 1.13
    from tensorflow.python.framework.tensor_util import is_tensor

__all__ = [
    "ste_sign",
    "swish_sign",
    "dorefa_quantizer",
    "SteSign",
    "SwishSign",
    "DoReFaQuantizer",
]


def _clipped_gradient(x, dy, clip_value):
    if clip_value is None:
        return dy

    zeros = tf.zeros_like(dy)
    mask = tf.math.less_equal(tf.math.abs(x), clip_value)
    return tf.where(mask, dy, zeros)


class QuantizerFunctionWrapper:
    """Wraps a quantizer function in a class that can be serialized.

    # Arguments
    fn: The quantizer function to wrap, with signature `fn(x, **kwargs)`.
    **kwargs: The keyword arguments that are passed on to `fn`.
    """

    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.precision = getattr(fn, "precision", 32)
        self._fn_kwargs = kwargs

    def __call__(self, x):
        """Invokes the `QuantizerFunctionWrapper` instance.

        # Arguments
        x: Input tensor.

        # Returns
        Quantized tensor.
        """
        return self.fn(x, **self._fn_kwargs)

    def get_config(self):
        return {
            k: tf.keras.backend.eval(v)
            if is_tensor(v) or isinstance(v, tf.Variable)
            else v
            for k, v in self._fn_kwargs.items()
        }


@utils.register_keras_custom_object
@utils.set_precision(1)
def ste_sign(x, clip_value=1.0):
    r"""Sign binarization function.

    \\[
    q(x) = \begin{cases}
      -1 & x < 0 \\\
      1 & x \geq 0
    \end{cases}
    \\]

    The gradient is estimated using the Straight-Through Estimator
    (essentially the binarization is replaced by a clipped identity on the
    backward pass).
    \\[\frac{\partial q(x)}{\partial x} = \begin{cases}
      1 & \left|x\right| \leq \texttt{clip_value} \\\
      0 & \left|x\right| > \texttt{clip_value}
    \end{cases}\\]

    ```plot-activation
    quantizers.ste_sign
    ```

    # Arguments
    x: Input tensor.
    clip_value: Threshold for clipping gradients. If `None` gradients are not clipped.

    # Returns
    Binarized tensor.

    # References
    - [Binarized Neural Networks: Training Deep Neural Networks with Weights and
      Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830)
    """

    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value)

        return math.sign(x), grad

    return _call(x)


def _scaled_sign(x):  # pragma: no cover
    return 1.3 * ste_sign(x)


@utils.register_keras_custom_object
@utils.set_precision(1)
def swish_sign(x, beta=5.0):
    r"""Sign binarization function.

    \\[
    q(x) = \begin{cases}
      -1 & x < 0 \\\
      1 & x \geq 0
    \end{cases}
    \\]

    The gradient is estimated using the SignSwish method.

    \\[
    \frac{\partial q_{\beta}(x)}{\partial x} = \frac{\beta\left\\{2-\beta x \tanh \left(\frac{\beta x}{2}\right)\right\\}}{1+\cosh (\beta x)}
    \\]

    ```plot-activation
    quantizers.swish_sign
    ```
    # Arguments
    x: Input tensor.
    beta: Larger values result in a closer approximation to the derivative of the sign.

    # Returns
    Binarized tensor.

    # References
    - [BNN+: Improved Binary Network Training](https://arxiv.org/abs/1812.11800)
    """

    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            b_x = beta * x
            return dy * beta * (2 - b_x * tf.tanh(b_x * 0.5)) / (1 + tf.cosh(b_x))

        return math.sign(x), grad

    return _call(x)


@utils.register_keras_custom_object
@utils.set_precision(2)
def dorefa_quantizer(x, k_bit=2):
    r"""k_bit quantizer as in the DoReFa paper.

    \\[
    q(x) = \begin{cases}
    0 & x < \frac{1}{2n} \\\
    \frac{i}{n} & \frac{2i-1}{2n} < |x| < \frac{2i+1}{2n} \text{ for } i \in \\{1,n-1\\}\\\
     1 & \frac{2n-1}{2n} < x
    \end{cases}
    \\]

    where \\(n = 2^{\text{k_bit}} - 1\\). The number of bits, k_bit, needs to be passed as an argument.
    The gradient is estimated using the Straight-Through Estimator
    (essentially the binarization is replaced by a clipped identity on the
    backward pass).
    \\[\frac{\partial q(x)}{\partial x} = \begin{cases}
    1 &  0 \leq x \leq 1 \\\
    0 & \text{else}
    \end{cases}\\]

    ```plot-activation
    quantizers.dorefa_quantizer
    ```

    # Arguments
    k_bit: number of bits for the quantization.

    # Returns
    quantized tensor

    # References
    - [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks
      with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
    """
    x = tf.clip_by_value(x, 0.0, 1.0)

    @tf.custom_gradient
    def _k_bit_with_identity_grad(x):
        n = 2 ** k_bit - 1
        return tf.round(x * n) / n, lambda dy: dy

    return _k_bit_with_identity_grad(x)


@utils.register_keras_custom_object
class SteSign(QuantizerFunctionWrapper):
    r"""Instantiates a serializable binary quantizer.

    \\[
    q(x) = \begin{cases}
      -1 & x < 0 \\\
      1 & x \geq 0
    \end{cases}
    \\]

    The gradient is estimated using the Straight-Through Estimator
    (essentially the binarization is replaced by a clipped identity on the
    backward pass).
    \\[\frac{\partial q(x)}{\partial x} = \begin{cases}
      1 & \left|x\right| \leq \texttt{clip_value} \\\
      0 & \left|x\right| > \texttt{clip_value}
    \end{cases}\\]

    ```plot-activation
    quantizers.ste_sign
    ```

    # Arguments
    clip_value: Threshold for clipping gradients. If `None` gradients are not clipped.

    # References
    - [Binarized Neural Networks: Training Deep Neural Networks with Weights and
      Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830)
    """

    def __init__(self, clip_value=1.0):
        super().__init__(ste_sign, clip_value=clip_value)



@utils.register_keras_custom_object
class SwishSign(QuantizerFunctionWrapper):
    r"""Sign binarization function.

    \\[
    q(x) = \begin{cases}
      -1 & x < 0 \\\
      1 & x \geq 0
    \end{cases}
    \\]

    The gradient is estimated using the SignSwish method.

    \\[
    \frac{\partial q_{\beta}(x)}{\partial x} = \frac{\beta\left\\{2-\beta x \tanh \left(\frac{\beta x}{2}\right)\right\\}}{1+\cosh (\beta x)}
    \\]

    ```plot-activation
    quantizers.swish_sign
    ```
    # Arguments
    beta: Larger values result in a closer approximation to the derivative of the sign.

    # Returns
    SwishSign quantization function

    # References
    - [BNN+: Improved Binary Network Training](https://arxiv.org/abs/1812.11800)
    """

    def __init__(self, beta=5.0):
        super().__init__(swish_sign, beta=beta)



@utils.register_keras_custom_object
class DoReFaQuantizer(QuantizerFunctionWrapper):
    r"""Instantiates a serializable k_bit quantizer as in the DoReFa paper.

    \\[
    q(x) = \begin{cases}
    0 & x < \frac{1}{2n} \\\
    \frac{i}{n} & \frac{2i-1}{2n} < |x| < \frac{2i+1}{2n} \text{ for } i \in \\{1,n-1\\}\\\
     1 & \frac{2n-1}{2n} < x
    \end{cases}
    \\]

    where \\(n = 2^{\text{k_bit}} - 1\\). The number of bits, k_bit, needs to be passed as an argument.
    The gradient is estimated using the Straight-Through Estimator
    (essentially the binarization is replaced by a clipped identity on the
    backward pass).
    \\[\frac{\partial q(x)}{\partial x} = \begin{cases}
    1 &  0 \leq x \leq 1 \\\
    0 & \text{else}
    \end{cases}\\]

    ```plot-activation
    quantizers.dorefa_quantizer
    ```

    # Arguments
    k_bit: number of bits for the quantization.

    # Returns
    Quantization function

    # References
    - [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks
      with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
    """

    def __init__(self, k_bit):
        super().__init__(dorefa_quantizer, k_bit=k_bit)
        self.precision = k_bit


def serialize(quantizer):
    return tf.keras.utils.serialize_keras_object(quantizer)


def deserialize(name, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="quantization function",
    )


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if isinstance(identifier, str):
        return deserialize(str(identifier))
    if callable(identifier):
        return identifier
    raise ValueError(
        f"Could not interpret quantization function identifier: {identifier}"
    )
