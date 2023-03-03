import tensorflow as tf


class L1pmRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(
        self,
        lmbda: float = 0.0,
    ):
        self.lmbda = lmbda

    def __call__(self, w):
        # TODO
        return self.lmbda

    def get_config(self):
        return {"l1": float(self.l1)}
