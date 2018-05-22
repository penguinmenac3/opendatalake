import os
import numpy as np
from scipy.misc import imread

from opendatalake.detection.utils import Detection25d, Detection2d, apply_projection, vec_len

PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


def _gen(params, stride=1, offset=0, infinite=False):
    images, calibrations, data_split, phase, data = params

    loop_condition = True
    while loop_condition:
        for idx in range(offset, len(images), stride):
            if data_split and idx % data_split == 0 and phase == PHASE_TRAIN:
                continue

            if data_split and idx % data_split != 0 and phase == PHASE_VALIDATION:
                continue

            detections2d = []
            detections25d = []

            for anno in data[idx]:
                date = anno.split(" ")
                if int(date[2]) == -1:
                    continue

                rect = [float(x) for x in date[4:8]]
                detections2d.append(Detection2d(class_id=date[0],
                                                cx=(rect[0] + rect[2]) / 2.0, cy=(rect[1] - rect[3]) / 2.0,
                                                w=rect[0] - rect[2], h=rect[1] - rect[3]))

                # Calc cx and cy in image coordinates.
                translation = [float(x) for x in date[11:14]]
                center = np.array(translation) + np.array([0.0, -float(date[8]) / 2.0, 0])
                projected_center = apply_projection(center, calibrations[idx])
                dist = vec_len(center)
                detections25d.append(Detection25d(class_id=date[0],
                                                    cx=projected_center[0][0], cy=projected_center[1][0], dist=dist,
                                                    w=float(date[9]), h=float(date[8]), l=float(date[10]),
                                                    theta=float(date[3])))

            feature = imread(images[idx], mode="RGB")
            yield ({"image": feature, "calibration": calibrations[idx]},
                   {"detections_2d": detections2d, "detections_2.5d": detections25d})
        loop_condition = infinite


def kitti_detection(base_dir, phase, data_split=10):
    images = []
    calibrations = []
    features = []

    for filename in os.listdir(os.path.join(base_dir, "training", "label_2")):
        if filename.endswith(".txt"):
            images.append(os.path.join(base_dir, "data_object_image_2", "training", "image_2", filename.replace(".txt", ".png")))

            calibration_file = os.path.join(base_dir, "data_object_calib", "training", "calib", filename)
            calibration_matrix = np.genfromtxt(calibration_file, delimiter=' ', usecols=range(1, 13), skip_footer=3)
            calibration_matrix = calibration_matrix[2].reshape(3, 4)
            calibrations.append(calibration_matrix)

            with open(os.path.join(base_dir, "training", "label_2", filename), 'r') as myfile:
                data = myfile.read().strip().split("\n")
                # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
                features.append(data)

    return _gen, (images, calibrations, data_split, phase, features)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Loading Dataset:")
    train_data = kitti_detection("datasets/kitti_detection", phase=PHASE_TRAIN)

    data_fn, data_params = train_data
    data_gen = data_fn(data_params)

    for feat, label in data_gen:
        img = feat["image"].copy()
        for detection in label["detections_2.5d"]:
            detection.visualize(img, color=(255, 0, 0), projection_matrix=feat["calibration"])
        plt.imshow(img)
        plt.show()
