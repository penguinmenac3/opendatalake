import os
import math
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
                                                cx=(rect[0] + rect[2]) / 2.0, cy=(rect[1] + rect[3]) / 2.0,
                                                w=rect[2] - rect[0], h=rect[3] - rect[1]))

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


def evaluate3d(predictor, prediction_2_detections, base_dir):
    test_data = kitti_detection(base_dir, phase=PHASE_VALIDATION)
    data_fn, data_params = test_data
    data_gen = data_fn(data_params)
    n = 100
    treshs = [i / float(n) for i in range(n + 1)]

    recalls = {}
    s_rs = {}

    for tresh in treshs:
        recalls[tresh] = []
        s_rs[tresh] = []

    for feat, label in data_gen:
        calib = feat["calibration"]
        gts = label["detection_2.5d"]
        predictor_output = predictor(feat)
        for tresh in treshs:
            preds = prediction_2_detections(predictor_output, tresh)
            TP, FP, FN = _optimal_assign(preds, gts, calib)
            recall = len(TP) / (len(TP) + len(FN))
            s_r = 0
            for p in TP:
                s_r += (1.0 + math.cos(p.a.theta - p.b.theta)) / 2.0
            s_r *= 1.0 / len(preds)
            recalls[tresh].append(recall)
            s_rs[tresh].append(s_r)

    # Compute mean recalls and mean s_rs
    for tresh in treshs:
        recalls[tresh] = sum(recalls[tresh]) / float(len(recalls[tresh]))
        s_rs[tresh] = sum(s_rs[tresh]) / float(len(s_rs[tresh]))

    aos = 0
    optimal_tresh = {}
    for r in [i / float(10) for i in range(11)]:
        max_s_r = 0
        for tresh in treshs:
            if recalls[tresh] >= r:
                if max_s_r < s_rs[tresh]:
                    max_s_r = s_rs[tresh]
                    optimal_tresh[r] = tresh
        aos += max_s_r
    aos *= 1 / 11

    return aos, optimal_tresh


def _optimal_assign(preds, gts, projection_matrix, tresh=0.5):
    TP, FP, FN = [], [], []

    class matching(object):
        def __init__(self, iou, a, b):
            self.iou = iou
            self.a = a
            self.b = b

        def __lt__(self, other):
            return self.iou < other.iou

        def __gt__(self, other):
            return self.iou > other.iou

    matches = []
    for p in preds:
        for g in gts:
            iou = p.iou(g, projection_matrix=projection_matrix)
            if iou > tresh:
                matches.append(matching(iou, p, g))

    matches = sorted(matches)

    assigned = []
    for m in matches:
        # Check if a or b have already matched better to something else
        if m.a in assigned or m.b in assigned:
            continue

        # It is the best match for this match
        assigned.append(m.a)
        assigned.append(m.b)
        TP.append(m)

    # All unassigned predictions are false positives.
    for a in preds:
        if a not in assigned:
            FP.append(a)

    # All unassigned ground truths are false negatives.
    for b in gts:
        if b not in assigned:
            FN.append(b)

    return TP, FP, FN


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
