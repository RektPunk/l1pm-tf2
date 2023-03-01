import tensorflow as tf
import numpy as np
from l1pm import (
    generate_example,
    DataTransformer,
    L1pm,
)

# Examples setting
EPOCHS = 1000
LEARNING_RATE = 0.005
N_SAMPLES = 100
TAUS = [0.1, 0.5, 0.9]
OUT_FEATURES = 10
DENSE_FEATURES = 10

x_train, y_train = generate_example(N_SAMPLES)
taus = np.array(TAUS)

data_transformer = DataTransformer(
    x=x_train,
    taus=taus,
    y=y_train,
)

x_train_transform, y_train_transform, taus_transform = data_transformer()

l1pm_regressor = L1pm(
    out_features=OUT_FEATURES,
    dense_features=DENSE_FEATURES,
    activation=tf.nn.sigmoid,
    n_taus=len(taus),
)

y_hat = l1pm_regressor(x_train)
