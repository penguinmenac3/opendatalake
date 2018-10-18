import os
import pickle
import numpy as np

from opendatalake.simple_sequence import SimpleSequence


# Here the cifar data can be downloaded.
CIFAR_10_DOWNLOAD = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_100_DOWNLOAD = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
# Extract it into the data/cifar-x folder
# For cifar-10 the data/cifar-10 folder should contain a data_batch_1, ... and a test_batch file.
# For cifar-100 the data/cifar-100 folder should contain a train and a test file.


class Cifar(SimpleSequence):
    def __init__(self, hyperparams, phase, preprocess_fn=None, augmentation_fn=None):
        super(Cifar, self).__init__(hyperparams, phase, preprocess_fn, augmentation_fn)
        base_dir = self.hyperparams.problem.data_path
        version = self.hyperparams.problem.get("version", 10)

        self.images = []
        self.labels = []

        if version == 10:
            if phase == "train":
                for x in range(1,5,1):
                    data_path = os.path.join(base_dir, "data_batch_" + str(x))
                    with open(data_path, 'rb') as fo:
                        dict = pickle.load(fo, encoding='bytes')
                        self.images.extend(dict[b"data"])
                        self.labels.extend(dict[b"labels"])
            if phase == "test":
                data_path = os.path.join(base_dir, "test_batch")
                with open(data_path, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    self.images.extend(dict[b"data"])
                    self.labels.extend(dict[b"labels"])
        if version == 100:
            data_path = os.path.join(base_dir, phase)
            with open(data_path, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.images.extend(dict[b"data"])
                self.labels.extend(dict[b"fine_labels"])

    def __num_samples(self):
        return len(self.images)

    def __get_sample(self, idx):
        img = np.reshape(self.images[idx], (3, 32, 32))
        return ({"image": img.transpose((1, 2, 0))}, {"probs": self.labels[idx]})
