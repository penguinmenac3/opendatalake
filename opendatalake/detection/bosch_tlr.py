import os
import sys
import yaml

import numpy as np
from scipy.misc import imread

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


def _gen(params, stride=1, offset=0, infinite=False):
    data = params

    loop_condition = True
    while loop_condition:
        for idx in range(offset, len(data), stride):
            date = data[idx]
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

            yield ({"image": feature},
                   {"detections_2d": detections2d})
        loop_condition = infinite


def bosch_tlr(base_dir, phase, riib=False):
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

    return _gen, images


if __name__ == "__main__":
    import cv2

    print("Loading Dataset:")
    train_data = bosch_tlr("datasets/bosch_tlr", phase=PHASE_TRAIN)

    data_fn, data_params = train_data
    data_gen = data_fn(data_params)

    for feat, label in data_gen:
        img = feat["image"].copy()
        if len(label["detections_2d"]) < 1:
            continue
        for detection in label["detections_2d"]:
            color = COLOR_MAP["Ignore"]
            if detection.class_id in COLOR_MAP:
                color = COLOR_MAP[detection.class_id]
            else:
                print("Unknown color: {}".format(detection.class_id))
            detection.visualize(img, color=color)
        cv2.imshow("Bosch TLR", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(500)
