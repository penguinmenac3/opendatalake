import os
import sys
import time
import math
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
try:
    from IPython.display import clear_output
    NO_IPYTHON = False
except ModuleNotFoundError:
    NO_IPYTHON = True

from opendatalake.detection.utils import Detection25d, Detection2d, apply_projection, vec_len

PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


def _gen(params, stride=1, offset=0, infinite=False):
    filenames, data_split, phase, base_dir = params

    loop_condition = True
    while loop_condition:
        for idx in range(offset, len(filenames), stride):
            filename = filenames[idx]
            image = os.path.join(base_dir, "data_object_image_2", "training", "image_2",
                                       filename.replace(".txt", ".png"))

            calibration_file = os.path.join(base_dir, "data_object_calib", "training", "calib", filename)
            calibration = np.genfromtxt(calibration_file, delimiter=' ', usecols=range(1, 13), skip_footer=3)
            calibration = calibration[2].reshape(3, 4)

            with open(os.path.join(base_dir, "training", "label_2", filename), 'r') as myfile:
                data = myfile.read().strip().split("\n")
                # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01

            if data_split and idx % data_split == 0 and phase == PHASE_TRAIN:
                continue

            if data_split and idx % data_split != 0 and phase == PHASE_VALIDATION:
                continue

            detections2d = []
            detections25d = []

            for anno in data:
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
                projected_center = apply_projection(center, calibration)
                dist = vec_len(center)
                detections25d.append(Detection25d(class_id=date[0],
                                                    cx=projected_center[0][0], cy=projected_center[1][0], dist=dist,
                                                    w=float(date[9]), h=float(date[8]), l=float(date[10]),
                                                    theta=float(date[3])))

            feature = imread(image, mode="RGB")
            yield ({"image": feature, "calibration": calibration},
                   {"detections_2d": detections2d, "detections_2.5d": detections25d})
        loop_condition = infinite


def kitti_detection(base_dir, phase, data_split=10):
    filenames = [f for f in os.listdir(os.path.join(base_dir, "training", "label_2")) if f.endswith(".txt")]

    return _gen, (filenames, data_split, phase, base_dir)


def evaluate3d(predictor, prediction_2_detections, base_dir, visualize=False, inline_plotting=False, img_path_prefix=None, min_tresh=0.5, steps=11):
    if NO_IPYTHON:
        print("Inline plotting not availible. Could not find ipython clear_output")
        inline_plotting = False
    print("Loading Data.")
    test_data = kitti_detection(base_dir, phase=PHASE_VALIDATION)
    data_fn, data_params = test_data
    data_gen = data_fn(data_params)
    treshs = [min_tresh + i / float(steps - 1) * (1.0 - min_tresh) for i in range(steps)]

    recalls = {}
    s_rs = {}

    for tresh in treshs:
        recalls[tresh] = []
        s_rs[tresh] = []

    print("Evaluating Samples")
    i = 0
    for feat, label in data_gen:
        if inline_plotting and i > 0:
            clear_output()
        print("Sample {}\r".format(i))
        i += 1
        calib = feat["calibration"]
        gts = label["detections_2.5d"]
        start = time.time()
        predictor_output = predictor(feat["image"])
        prediction_time = time.time() - start
        for tresh in treshs:
            start = time.time()
            preds = prediction_2_detections(predictor_output, tresh, calib)
            conversion_time = time.time() - start
            TP, FP, FN = _optimal_assign(preds, gts, calib)
            recall = len(TP) / (len(TP) + len(FN))
            s_r = 0
            for p in TP:
                s_r += (1.0 + math.cos(p.a.theta - p.b.theta)) / 2.0
            normalizer = len(preds)
            if normalizer == 0:
                s_r = 0
                print("Warn: No preds!")
                # FIXME is this a good idea?
            else:
                s_r *= 1.0 / normalizer
            recalls[tresh].append(recall)
            s_rs[tresh].append(s_r)

            if visualize:
                print("TP {} FP {} FN {} Tresh {:.2f} CNN {:.3f}s Postprocessing {:.3f}s".format(len(TP), len(FP), len(FN), tresh, prediction_time, conversion_time))
                image = feat["image"].copy()
                for match in TP:
                    match.b.visualize(image, (0, 255, 255), projection_matrix=calib)
                    match.a.visualize(image, (0, 255, 0), projection_matrix=calib)
                for a in FP:
                    a.visualize(image, (255, 0, 0), projection_matrix=calib)
                for b in FN:
                    b.visualize(image, (128, 0, 0), projection_matrix=calib)
                plt.clf()
                plt.title("TP {} FP {} FN {} Tresh {:.2f}".format(len(TP), len(FP), len(FN), tresh))
                plt.imshow(image)
                img_path = "images/{:04d}_{:.2f}.png".format(i, tresh)
                if img_path_prefix is not None:
                    img_path = os.path.join(img_path_prefix, img_path)
                plt.savefig(img_path)
                plt.show()
        
        if inline_plotting and i > 0:
            plt.clf()
            plt.title("Recall Curve")
            plt.xlabel("Treshs")
            plt.ylabel("Recall")
            plt.plot(treshs, [sum(recalls[t])/float(len(recalls[t])) for t in treshs])
            plt.show()

    print("Computing AOS.")
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

    print("Computing AOS done.")
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
    from opendatalake.texture_augmentation import full_texture_augmentation

    print("Loading Dataset:")
    train_data = kitti_detection("datasets/kitti_detection", phase=PHASE_TRAIN)

    data_fn, data_params = train_data
    data_gen = data_fn(data_params)

    for feat, label in data_gen:
        img = feat["image"].copy()
        img = full_texture_augmentation(img)
        for detection in label["detections_2.5d"]:
            detection.visualize(img, color=(255, 0, 0), projection_matrix=feat["calibration"])
        plt.imshow(img)
        plt.show()
