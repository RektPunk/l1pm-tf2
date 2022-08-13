#%%
import numpy as np
import tensorflow as tf


class l1pmOutputDense(tf.keras.layers.Layer):
    """
    Mcqrnn Output dense network
    *Note that the activation of the last hidden layer must be sigmoid.
    **Or, Monotone increasing function bounded by [0, 1].
    Methods:
        build:
            Set weight shape for first call
        call:
            Return dense layer with activation
    """

    def __init__(
        self,
        n_taus: int,
        kernel_regularizer: tf.keras.regularizers.Regularizer = None,
        **kwargs,
    ):
        super(l1pmOutputDense, self).__init__(**kwargs)
        self.n_taus = n_taus
        self.kernel_regularizer = kernel_regularizer

    def build(
        self,
        input_shape,
    ):
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.n_taus]), name="w")
        self.b = tf.Variable(tf.zeros([self.n_taus]), name="b")
        _w_concat = tf.concat([self.b, self.w], axis=0)
        self.w_cumsum = tf.cumsum(_w_concat)[1:]
        self.kernel = self.add_weight(
            "l1pmkernel",
            shape=[input_shape[-1] + 1, self.n_taus],
            regularizer=self.kernel_regularizer,
        )

    def call(
        self,
        inputs: np.ndarray,
    ):
        outputs = tf.matmul(inputs, self.w_cumsum) + self.b
        outputs = tf.reshape(outputs, [-1, 1])
        return outputs
