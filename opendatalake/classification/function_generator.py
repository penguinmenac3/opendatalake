from random import shuffle, randint, random
import numpy as np

from opendatalake.simple_sequence import SimpleSequence
from opendatalake.utils import one_hot


class FunctionGenerator(SimpleSequence):
    def __init__(self, hyperparams, phase, functions, preprocess_fn=None, augmentation_fn=None):
        super(FunctionGenerator, self).__init__(hyperparams, phase, preprocess_fn, augmentation_fn)
        self.functions = functions
        self.num_functions = len(functions)
        self.sequence_length = hyperparams.problem.sequence_length
        self.samples_per_epoch = hyperparams.problem.samples_per_epoch

    def __num_samples(self):
        return self.samples_per_epoch

    def __get_sample(self, idx):
        label_idx = randint(0, self.num_functions - 1)
        offset = random()
        selected_function = self.functions[label_idx]
        feature = np.array([selected_function(i, offset) for i in range(self.sequence_length)], dtype=np.float32)
        yield {"feature": feature}, {"probs": one_hot(label_idx, self.num_functions)}
