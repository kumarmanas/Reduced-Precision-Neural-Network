import logging
import tensorflow as tf
from larq import quantizers, utils, metrics as lq_metrics

log = logging.getLogger(__name__)


# TODO: find a good way remove duplication between QuantizerBase, QuantizerDepthwiseBase and QuantizerSeparableBase


class QuantizerBase(tf.keras.layers.Layer):
    """Base class for defining quantized layers

    `input_quantizer` and `kernel_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Layer`.
    """

    def __init__(
        self, *args, input_quantizer=None, kernel_quantizer=None, metrics=None, **kwargs
    ):
        self.input_quantizer = quantizers.get(input_quantizer)
        self.kernel_quantizer = quantizers.get(kernel_quantizer)
        self.quantized_latent_weights = []
        self.quantizers = []
        self._custom_metrics = (
            metrics if metrics is not None else lq_metrics.get_training_metrics()
        )

        super().__init__(*args, **kwargs)
        if kernel_quantizer and not self.kernel_constraint:
            log.warning(
                "Using a weight quantizer without setting `kernel_constraint` "
                "may result in starved weights (where the gradient is always zero)."
            )

    def build(self, input_shape):
        super().build(input_shape)
        if self.kernel_quantizer:
            self.quantized_latent_weights.append(self.kernel)
            self.quantizers.append(self.kernel_quantizer)
            if "flip_ratio" in self._custom_metrics and utils.supports_metrics():
                self.flip_ratio = lq_metrics.FlipRatio(
                    values_shape=self.kernel.shape, name=f"flip_ratio/{self.name}"
                )

    @property
    def non_trainable_weights(self):
        weights = super().non_trainable_weights
        if hasattr(self, "flip_ratio"):
            return [
                weight
                for weight in weights
                if not any(weight is metric_w for metric_w in self.flip_ratio.weights)
            ]
        return weights

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)

        with utils.quantize(self, "kernel", self.kernel_quantizer) as kernel:
            if hasattr(self, "flip_ratio"):
                self.add_metric(self.flip_ratio(kernel))
            return super().call(inputs)

    def get_config(self):
        config = {
            "input_quantizer": quantizers.serialize(self.input_quantizer),
            "kernel_quantizer": quantizers.serialize(self.kernel_quantizer),
        }
        return {**super().get_config(), **config}


