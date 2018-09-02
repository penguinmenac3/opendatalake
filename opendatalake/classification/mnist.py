import math
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import Sequence


class MNIST(Sequence):
    def __init__(self, hyperparams, phase, preprocess_feature=None, preprocess_label=None, augment_data=None):
        _mnist = input_data.read_data_sets(hyperparams.problem.data_path, one_hot=True)
        if phase == "validation":
            images = _mnist.test.images
            labels = _mnist.test.labels
        else:
            images = _mnist.train.images
            labels = _mnist.train.labels

        self.images = images
        self.labels = labels
        self.hyperparams = hyperparams
        self.batch_size = hyperparams["train"]["batch_size"]
        self.preprocess_feature = preprocess_feature
        self.preprocess_label = preprocess_label
        self.augment_data = augment_data

    def __len__(self):
        return math.ceil(len(self.images)/self.batch_size)

    def __getitem__(self, index):
        features = []
        labels = []
        for idx in range(index * self.batch_size, min((index + 1) * self.batch_size, len(self))):
            feature = {"image": np.reshape(self.images[idx], (28, 28))}
            label = {"probs": self.labels[idx]}
            if self.augment_data is not None:
                feature, label = self.augment_data(self.hyperparams, feature, label)
            if self.preprocess_feature is not None:
                feature = self.preprocess_feature(self.hyperparams, feature)
            if self.preprocess_label is not None:
                label = self.preprocess_label(self.hyperparams, feature, label)
            features.append(feature)
            labels.append(label)
        return {k: np.array([dic[k] for dic in features]) for k in features[0]},\
               {k: np.array([dic[k] for dic in labels]) for k in labels[0]}
