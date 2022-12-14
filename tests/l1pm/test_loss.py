import numpy as np
from l1pm.loss import TiltedAbsoluteLoss


def test_tilted_absolute_loss():
    tilted_absolute_loss = TiltedAbsoluteLoss(tau=np.array([0.1], dtype="float32"))
    loss = tilted_absolute_loss(
        y_true=np.array([10, 2, 4, 3], dtype="float32"),
        y_pred=np.array([3, 2, 1, 7], dtype="float32"),
    )

    assert round(loss.numpy().astype("float"), 8) == 1.14999998
