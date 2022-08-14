from typing import Callable
import numpy as np
import tensorflow as tf
from l1pm.layers import l1pmOutputDense


class l1pm(tf.keras.Model):
    """
    l1pm simple structure
    Note that the middle of dense network can be modified
    Args:
        out_features (int): the number of nodes in first hidden layer
        dense_features (int): the number of nodes in hidden layer
        activation (Callable): activation function e.g. tf.nn.relu or tf.nn.sigmoid
    Methods:
        call:
            Return dense layer with input activation and features
    """

    def __init__(
        self,
        out_features: int,
        dense_features: int,
        n_taus: int, 
        activation: Callable = tf.nn.sigmoid,
        **kwargs,
    ):
        super(l1pm, self).__init__(**kwargs)
        self.n_taus = n_taus
        self.input_dense = tf.keras.layers.Dense(
            out_features=out_features,
            activation=activation,
        )
        self.dense = tf.keras.layes.Dense(
            dense_features=dense_features,
            activation=tf.nn.sigmoid,
        )
        self.output_dense = l1pmOutputDense(
            n_taus= self.n_taus,
        )

    def call(
        self,
        inputs: np.ndarray,
    ):
        x = self.input_dense(inputs)
        x = self.dense(x)
        outputs = self.output_dense(x)
        return outputs
