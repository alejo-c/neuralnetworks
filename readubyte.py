import sys
import pickle
import numpy as np


def vectorized_result(digit):
    return np.eye(10)[digit]


with open(sys.argv[1], "rb") as file:
    training_data, validation_data, test_data = pickle.loads(
        file.read(), encoding="latin-1"
    )

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]

    # training_inputs = np.array(training_inputs)[:784].reshape(28, 28)
    # training_results = np.array(training_results)[:784].reshape(28, 28)
    training_inputs = np.array(training_inputs)
    training_results = np.array(training_results)
