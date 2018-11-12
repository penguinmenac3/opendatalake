import os
from scipy.misc import imread
import json

from opendatalake.simple_sequence import SimpleSequence
from opendatalake.utils import crop_center


class UnlabeledImageFolder(SimpleSequence):
    def __init__(self, hyperparams, phase, preprocess_fn=None, augmentation_fn=None):
        super(UnlabeledImageFolder, self).__init__(hyperparams, phase, preprocess_fn, augmentation_fn)

        self.base_dir = self.hyperparams.problem.data_path
        self.crop_roi = self.hyperparams.problem.get("crop_roi", None)
        file_extension = self.hyperparams.problem.get("file_extension", ".png")

        if phase is not None:
            data_dir = os.path.join(self.base_dir, phase)
        else:
            data_dir = self.base_dir
        self.images = []

        for filename in os.listdir(data_dir):
            if filename.endswith(file_extension):
                self.images.append(os.path.join(data_dir, filename))


    def num_samples(self):
        return len(self.images)

    def get_sample(self, idx):
        feature = imread(self.images[idx], mode="RGB")
        if self.crop_roi is not None:
            feature = crop_center(feature, self.crop_roi[0], self.crop_roi[1])
        label = feature.copy()
        return ({"image": feature}, {"image": label})
