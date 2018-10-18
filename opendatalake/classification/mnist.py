import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from opendatalake.simple_sequence import SimpleSequence


class MNIST(SimpleSequence):
    def __init__(self, hyperparams, phase, preprocess_fn=None, augmentation_fn=None):
        super(MNIST, self).__init__(hyperparams, phase, preprocess_fn, augmentation_fn)
        _mnist = input_data.read_data_sets(hyperparams.problem.data_path, one_hot=True)
        if phase == "validation":
            images = _mnist.test.images
            labels = _mnist.test.labels
        else:
            images = _mnist.train.images
            labels = _mnist.train.labels

        self.images = images
        self.labels = labels

    def __num_samples(self):
        return len(self.images)

    def __get_sample(self, idx):
        return {"image": np.reshape(self.images[idx], (28, 28))}, {"probs": self.labels[idx]}
