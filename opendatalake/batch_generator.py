import numpy as np


def batch_generator(data, batch_size=0):
    """
        Create batches from a generator.
    """
    if batch_size <= 0:
        features = []
        labels = []
        data_gen = data[0](data[1])
        for feature, label in data_gen:
            features.append(feature)
            labels.append(label)
        out_tuple = np.array(features), np.array(labels)

        while True:
            yield out_tuple
    else:
        data_gen = data[0](data[1], infinite=True)
        while True:
            features = []
            labels = []
            for i in range(batch_size):
                feature, label = next(data_gen)
                features.append(feature)
                labels.append(label)
            yield np.array(features), np.array(labels)
