from math import ceil
from numpy import array
from keras.utils import Sequence


class SimpleSequence(Sequence):
    """
    This simple sequence makes implementing keras sequences in the correct way far simpler.

    A subclass must implement a __num_samples() -> int method
    and a __get_sample(idx: int) -> (feature_dict, label_dict) method which returns a single sample.

    This class automatically then applies augmentation, then preparation and finally batches as specified in
    hyperparams.train.batch_size.
    """
    def __init__(self, hyperparams, phase, preprocess_fn=None, augmentation_fn=None):
        self.hyperparams = hyperparams
        self.preprocess_fn = preprocess_fn
        self.augmentation_fn = augmentation_fn
        self.phase = phase

    def __num_samples(self):
        raise NotImplementedError("A subclass must implement this function to find out how many training samples it has.")

    def __get_sample(self, idx):
        raise NotImplementedError("A subclass must implement this. Returns a tuple of (feature, label) representing a single training sample.")

    def __len__(self):
        return ceil(self.__num_samples__() / self.hyperparams.train.get("batch_size", 1))

    def __getitem__(self, index):
        features = []
        labels = []
        batch_size = self.hyperparams.train.get("batch_size", 1)
        for idx in range(index * batch_size, min((index + 1) * batch_size, len(self.images))):
            feature, label = self.__get_sample(idx)
            if self.augmentation_fn is not None:
                feature, label = self.augmentation_fn(self.hyperparams, feature, label)
            if self.preprocess_fn is not None:
                feature, label = self.preprocess_fn(self.hyperparams, feature, label)
            features.append(feature)
            labels.append(label)
        return {k: array([dic[k] for dic in features]) for k in features[0]},\
               {k: array([dic[k] for dic in labels]) for k in labels[0]}
