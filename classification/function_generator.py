import os
from scipy.misc import imread
from random import shuffle, randint, random
import numpy as np
import json
import math


def one_hot(idx, max_idx):
    label = np.zeros(max_idx, dtype=np.uint8)
    label[idx] = 1
    return label


def function_generator(functions, sequence_length, num_datapoints, prepare_features=None):
    features = []
    labels = []
    num_functions = len(functions)

    def gen(skip_n=1, offset=0):
        for idx in range(offset, num_datapoints, skip_n):
            label_idx = randint(0, num_functions - 1)
            offset = random()
            selected_function = functions[label_idx]
            feature = np.array([selected_function(i, offset) for i in range(sequence_length)], dtype=np.float32)
            if prepare_features:
                feature = prepare_features(feature)
            yield feature, one_hot(label_idx, num_functions)

    return gen


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Loading Dataset:")
    train_data = function_generator([lambda x, off: math.sin(x / 50.0 + off), lambda x, off: x / 50.0 + off], 100, 10)()

    for feature, label in train_data:
        plt.plot(feature)
        plt.show()
