import os
import json
import numpy as np
from scipy.misc import imread

from opendatalake.simple_sequence import SimpleSequence
from opendatalake.utils import crop_center, one_hot


PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


class NamedFolders(SimpleSequence):
    def __init__(self, hyperparams, phase, preprocess_fn=None, augmentation_fn=None, overwrite_cache=False):
        super(NamedFolders, self).__init__(hyperparams, phase, preprocess_fn, augmentation_fn)

        base_dir = self.hyperparams.problem.data_path
        class_idx = self.hyperparams.problem.get("class_idx", {})
        crop_roi = self.hyperparams.problem.get("crop_roi", None)
        file_extension = self.hyperparams.problem.get("file_extension", ".png")
        validation_split = self.hyperparams.problem.get("validation_split", False)

        if validation_split:
            classes_dir = base_dir
        else:
            classes_dir = os.path.join(base_dir, phase)
        classes = [c for c in os.listdir(classes_dir) if c != "labels.json" and c != "images.json"]
        images = []
        labels = []

        if overwrite_cache:
            if os.path.exists(os.path.join(classes_dir, "images.json")):
                os.remove(os.path.join(classes_dir, "images.json"))
            if os.path.exists(os.path.join(classes_dir, "labels.json")):
                os.remove(os.path.join(classes_dir, "labels.json"))

        if os.path.exists(os.path.join(classes_dir, "images.json")) and os.path.exists(os.path.join(classes_dir, "labels.json")):
            print("Using buffer files.")
            with open(os.path.join(classes_dir, "images.json"), 'r') as infile:
                images = json.load(infile)
            with open(os.path.join(classes_dir, "labels.json"), 'r') as infile:
                labels = json.load(infile)
        else:
            print("No buffer files found. Reading folder structure and creating buffer files.")
            for c in classes:
                if c not in class_idx:
                    class_idx[c] = len(class_idx)
                class_dir = os.path.join(classes_dir, c)
                for filename in os.listdir(class_dir):
                    if filename.endswith(file_extension):
                        images.append(os.path.join(class_dir, filename))
                        labels.append(class_idx[c])

            with open(os.path.join(classes_dir, "images.json"), 'w') as outfile:
                json.dump(images, outfile)
            with open(os.path.join(classes_dir, "labels.json"), 'w') as outfile:
                json.dump(labels, outfile)

        self.n_classes = len(classes)
        self.images = images
        self.labels = labels
        self.crop_roi = crop_roi
        self.validation_split = validation_split

    def __num_samples(self):
        if validation_split:
            training_imgs = int(len(self.images) * (1.0 - validation_split))
            if self.phase == PHASE_VALIDATION:
                return len(self.images) - training_imgs
            else:
                return training_imgs
        else:
            return len(self.images)

    def __get_sample(self, idx):
        # Offset the validation images by the number of training images
        if validation_split and self.phase == PHASE_VALIDATION:
            idx = idx + int(len(self.images) * (1.0 - validation_split))
        feature = imread(self.images[idx], mode="RGB")
        if self.crop_roi is not None:
            feature = crop_center(feature, self.crop_roi[0], self.crop_roi[1])
        return ({"image": feature}, {"probs": one_hot(self.labels[idx], self.n_classes)})
