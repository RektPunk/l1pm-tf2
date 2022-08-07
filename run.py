import tensorflow as tf
import numpy as np
from l1pm import generate_example

# Examples setting
EPOCHS = 1000
LEARNING_RATE = 0.005
N_SAMPLES = 100
TAUS = [0.1, 0.5, 0.9]
OUT_FEATURES = 10
DENSE_FEATURES = 10

x_train, y_train = generate_example(N_SAMPLES)
taus = np.array(TAUS)
