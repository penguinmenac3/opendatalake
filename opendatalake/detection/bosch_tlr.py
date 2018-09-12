import os
import yaml
import math

import numpy as np
from scipy.misc import imread
import tensorflow as tf
Sequence = tf.keras.utils.Sequence

from opendatalake.detection.utils import Detection2d

PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"
COLOR_MAP = {
    "Background": (0, 0, 0),
    "Green": (255, 0, 0),
    "GreenRight": (255, 0, 0),
    "GreenLeft": (255, 0, 0),
    "Yellow": (255, 255, 0),
    "YellowRight": (255, 255, 0),
    "YellowLeft": (255, 255, 0),
    "Red": (0, 255, 0),
    "RedRight": (0, 255, 0),
    "RedLeft": (0, 255, 0),
    "off": (0, 128, 255),
    "Ignore": (0, 0, 255)
}


class BoschTLR(Sequence):
    def __init__(self, hyperparams, phase, riib=False, preprocess_feature=None, preprocess_label=None, augment_data=None):
        base_dir = hyperparams.problem.data_path
        input_yaml = "train.yaml" if phase == PHASE_TRAIN else "test.yaml"

        images = yaml.load(open(os.path.join(base_dir, input_yaml), 'rb').read())

        for i in range(len(images)):
            if phase == PHASE_VALIDATION:
                images[i]['path'] = os.path.join(base_dir, "rgb", "test", images[i]['path'].split("/")[-1])
            else:
                images[i]['path'] = os.path.join(base_dir, images[i]['path'])
            if riib:
                images[i]['path'] = images[i]['path'].replace('.png', '.pgm')
                images[i]['path'] = images[i]['path'].replace('rgb/train', 'riib/train')
                images[i]['path'] = images[i]['path'].replace('rgb/test', 'riib/test')
                for box in images[i]['boxes']:
                    box['y_max'] = box['y_max'] + 8
                    box['y_min'] = box['y_min'] + 8
        self.images = images
        self.hyperparams = hyperparams
        self.batch_size = hyperparams["train"]["batch_size"]
        self.preprocess_feature = preprocess_feature
        self.preprocess_label = preprocess_label
        self.augment_data = augment_data

    def __len__(self):
        return math.floor(len(self.images)/self.batch_size)

    def __getitem__(self, index):
        features = []
        labels = []
        for idx in range(index * self.batch_size, min((index + 1) * self.batch_size, len(self.images))):
            date = self.images[idx]
            feature = imread(date['path'], mode="RGB")
            detections2d = []

            for box in date['boxes']:
                class_id = box["label"]
                if box["occluded"]:
                    class_id = "Ignore"

                detections2d.append(Detection2d(class_id=class_id,
                                                cx=(box['x_min'] + box['x_max']) / 2.0,
                                                cy=(box['y_min'] + box['y_max']) / 2.0,
                                                w=box['x_max'] - box['x_min'],
                                                h=box['y_max'] - box['y_min']))
            feature_dict = None
            label_dict = None
            for i in range(10):
                feature_dict = {"image": feature}
                label_dict = {"detections_2d": detections2d}
                is_bad = False
                if self.augment_data is not None:
                    feature_dict, label_dict = self.augment_data(self.hyperparams, feature_dict, label_dict)
                if self.preprocess_feature is not None:
                    feature_dict, is_bad = self.preprocess_feature(self.hyperparams, feature_dict)
                if self.preprocess_label is not None and not is_bad:
                    label_dict, is_bad = self.preprocess_label(self.hyperparams, feature_dict, label_dict)
                if not is_bad:
                    break
            features.append(feature_dict)
            labels.append(label_dict)
        input_tensor_order = sorted(list(features[0].keys()))
        return {k: np.array([dic[k] for dic in features]) for k in input_tensor_order},\
               {k: np.array([dic[k] for dic in labels]) for k in labels[0]}
