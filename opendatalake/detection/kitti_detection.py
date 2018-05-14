import os
from scipy.misc import imread
import numpy as np
import json

PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


def one_hot(idx, max_idx):
    label = np.zeros(max_idx, dtype=np.uint8)
    label[idx] = 1
    return label


def _gen(params, stride=1, offset=0, infinite=False):
    images, calibrations, data_split, phase, class_ids, rects, boxes, occlusions = params

    loop_condition = True
    while loop_condition:
        for idx in range(offset, len(images), stride):
            if data_split and idx % data_split == 0 and phase == PHASE_TRAIN:
                continue

            if data_split and idx % data_split != 0 and phase == PHASE_VALIDATION:
                continue

            feature = imread(images[idx], mode="RGB")
            yield ({"image": feature, "calibration": calibrations[idx]}, {"class_id": class_ids[idx], "rect": rects[idx], "boxes": boxes[idx], "occlusion": occlusions[idx]})
        loop_condition = infinite


def named_folders(base_dir, phase, data_split=0.1):
    images = []
    calibrations = []
    class_ids = []
    rects = []
    boxes = []
    occlusions = []

    for filename in os.listdir(os.path.join(base_dir, "training", "label_2")):
        if filename.endswith(".txt"):
            # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
            images.append(os.path.join(base_dir, "data_object_image_2", "training", "image_2", filename.replace(".txt", ".png")))

            with open(os.path.join(base_dir, "data_object_calib", "training", "calib", filename), 'r') as myfile:
                calibrations.append(myfile.read())

            class_id = []
            rect = []
            box = []
            occlusion = []
            with open(os.path.join(base_dir, "training", "label_2", filename), 'r') as myfile:
                data = myfile.read().strip().split("\n")
                for anno in data:
                    date = anno.split(" ")
                    if int(date[2]) == -1:
                        continue
                    class_id.append(date[0])
                    occlusion.append(int(date[2]))
                    rect.append([float(x) for x in date[4:8]])
                    box.append({"alpha": float(date[3]),
                                "theta": float(date[14]),
                                "dimensions": [float(x) for x in date[8:11]],
                                "location": [float(x) for x in date[11:14]]})

            class_ids.append(class_id)
            rects.append(rect)
            boxes.append(box)
            occlusions.append(occlusion)

    return _gen, (images, calibrations, data_split, phase, class_ids, rects, boxes, occlusions)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Loading Dataset:")
    train_data = named_folders("../starttf/data/kitti_detection", phase=PHASE_TRAIN)

    data_fn, data_params = train_data
    data_gen = data_fn(data_params)

    feat, label = next(data_gen)
    print("Image shape:")
    print(feat["image"].shape)

    for feat, label in data_gen:
        print(label)
        print(feat["calibration"])
        plt.imshow(feat["image"])
        plt.show()
