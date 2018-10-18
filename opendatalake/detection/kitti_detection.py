import os
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

import tensorflow as tf
from opendatalake.detection.utils import Detection25d, Detection2d, apply_projection, vec_len

Sequence = tf.keras.utils.Sequence
PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


class KittiDetection(Sequence):
    def __init__(self, hyperparams, phase, preprocess_feature=None, preprocess_label=None, augment_data=None):
        data_split = hyperparams.problem.get("data_split", 10)
        depth_mapping_file_path = hyperparams.problem.get("depth_mapping_file_path", None)
        depth_base_dir = hyperparams.problem.get("depth_base_dir", None)
        base_dir = hyperparams.problem.data_path
        filenames = [f for f in os.listdir(os.path.join(base_dir, "training", "label_2")) if f.endswith(".txt")]
        load_depth = False
        if depth_mapping_file_path is not None and depth_base_dir is not None:
            mappings = []
            with open(depth_mapping_file_path, 'r') as myfile:
                mappings = myfile.read().strip().split("\n")
            load_depth = {}
            for mapping in mappings:
                same_files = mapping.split(" ")
                for f in filenames:
                    if os.path.join("data_object_image_2", "training", "image_2", f.replace(".txt", ".png")) == same_files[1]:
                        same_image_path = same_files[0]
                        path_parts = same_image_path.split("/")
                        drive = path_parts[1]
                        image_name = path_parts[-1]
                        depth_image_path = os.path.join(depth_base_dir, "train", drive, "proj_depth", "groundtruth", "image_02", image_name)
                        load_depth[f] = depth_image_path

        self.filenames = []
        for idx, filename in enumerate(filenames):
            if data_split and idx % data_split == 0 and phase == PHASE_TRAIN:
                continue

            if data_split and idx % data_split != 0 and phase == PHASE_VALIDATION:
                continue

            if not load_depth:
                pass
            else:
                if filename not in load_depth:
                    print("Image {} not in depth mapping.".format(filename))
                    continue
                if not os.path.exists(load_depth[filename]):
                    print("Image {} does not exist as depth image.".format(load_depth[filename]))
                    continue
            self.filenames.append(filename)

        self.base_dir = base_dir
        self.filenames = filenames
        self.phase = phase
        self.load_depth = load_depth
        self.hyperparams = hyperparams
        self.batch_size = hyperparams.train.batch_size
        self.preprocess_feature = preprocess_feature
        self.preprocess_label = preprocess_label
        self.augment_data = augment_data

    def __len__(self):
        return math.floor(len(self.filenames)/self.batch_size)

    def __getitem__(self, index):
        features = []
        labels = []
        for idx in range(index * self.batch_size, min((index + 1) * self.batch_size, len(self.filenames))):
            filename = self.filenames[idx]
            image = os.path.join(self.base_dir, "data_object_image_2", "training", "image_2",
                                 filename.replace(".txt", ".png"))

            calibration_file = os.path.join(self.base_dir, "data_object_calib", "training", "calib", filename)
            calibration = np.genfromtxt(calibration_file, delimiter=' ', usecols=range(1, 13), skip_footer=3)
            calibration = calibration[2].reshape(3, 4)

            with open(os.path.join(self.base_dir, "training", "label_2", filename), 'r') as myfile:
                data = myfile.read().strip().split("\n")
                # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01

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
            feature_dict = None
            label_dict = None
            for i in range(10):
                if not self.load_depth:
                    feature_dict = {"image": feature, "calibration": calibration}
                    label_dict = {"detections_2d": detections2d, "detections_2.5d": detections25d}
                else:
                    depth = imread(self.load_depth[filename])
                    feature_dict = {"image": feature, "calibration": calibration}
                    label_dict = {"detections_2d": detections2d, "detections_2.5d": detections25d, "depth": depth}

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


def evaluate3d(predictor, prediction_2_detections, hyperparams, visualize=False, inline_plotting=False, img_path_prefix=None, min_tresh=0.5, steps=11, allowed_classes=None):
    if NO_IPYTHON:
        print("Inline plotting not availible. Could not find ipython clear_output")
        inline_plotting = False
    print("Loading Data.")
    hyperparams.train.batch_size = 1  # Force batch size to 1
    test_data = KittiDetection(hyperparams, phase=PHASE_VALIDATION)
    treshs = [min_tresh + i / float(steps - 1) * (1.0 - min_tresh) for i in range(steps)]

    recalls = {}
    s_rs = {}

    for tresh in treshs:
        recalls[tresh] = []
        s_rs[tresh] = []

    print("Evaluating Samples")
    for i in range(len(test_data)):
        feat, label = test_data[i]
        feat = {k: feat[k][0] for k in list(feat.keys())}
        label = {k: label[k][0] for k in list(label.keys())}
        if inline_plotting and i > 0:
            clear_output()
        print("Sample {}\r".format(i))
        calib = feat["calibration"]
        gts = label["detections_2.5d"]
        if allowed_classes is not None:
            gts = [d for d in gts if d.class_id in allowed_classes]
        start = time.time()
        predictor_output = predictor(feat["image"], calib)
        prediction_time = time.time() - start
        for tresh in treshs:
            start = time.time()
            preds = prediction_2_detections(predictor_output, tresh, calib)
            conversion_time = time.time() - start
            TP, FP, FN = _optimal_assign(preds, gts, calib)
            recall = 1
            if len(TP) + len(FN) > 0:
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
                fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
                plt.title("TP {} FP {} FN {} Tresh {:.2f}".format(len(TP), len(FP), len(FN), tresh))
                plt.imshow(image)
                img_path = "images/{:04d}_{:.2f}.png".format(i, tresh)
                if img_path_prefix is not None:
                    img_path = os.path.join(img_path_prefix, img_path)
                plt.savefig(img_path)

                # Plot only preds
                image = feat["image"].copy()
                for match in TP:
                    match.a.visualize(image, (0, 255, 0), projection_matrix=calib)
                for a in FP:
                    a.visualize(image, (255, 0, 0), projection_matrix=calib)
                plt.clf()
                fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
                plt.title("TP {} FP {} Tresh {:.2f}".format(len(TP), len(FP), tresh))
                plt.imshow(image)
                img_path = "images/preds_{:04d}_{:.2f}.png".format(i, tresh)
                if img_path_prefix is not None:
                    img_path = os.path.join(img_path_prefix, img_path)
                plt.savefig(img_path)

                # Plot top down view.
                canvas = np.zeros(shape=(1000, 500, 3), dtype=np.uint8)
                for match in TP:
                    match.b.visualize_top_down(canvas, (0, 255, 255), projection_matrix=calib, scale=0.1)
                    match.a.visualize_top_down(canvas, (0, 255, 0), projection_matrix=calib, scale=0.1)
                for a in FP:
                    a.visualize_top_down(canvas, (255, 0, 0), projection_matrix=calib, scale=0.1)
                for b in FN:
                    b.visualize_top_down(canvas, (128, 0, 0), projection_matrix=calib, scale=0.1)
                plt.clf()
                fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
                plt.title("TP {} FP {} FN {} Tresh {:.2f}".format(len(TP), len(FP), len(FN), tresh))
                plt.imshow(canvas)
                img_path = "images/top_down_{:04d}_{:.2f}.png".format(i, tresh)
                if img_path_prefix is not None:
                    img_path = os.path.join(img_path_prefix, img_path)
                plt.savefig(img_path)

        if i > 0:
            plt.clf()
            plt.title("Recall Curve")
            plt.xlabel("Treshs")
            plt.ylabel("Recall")
            plt.plot(treshs, [sum(recalls[t])/float(len(recalls[t])) for t in treshs])
            img_path = "images/RecallCurve.png"
            if img_path_prefix is not None:
                img_path = os.path.join(img_path_prefix, img_path)
            plt.savefig(img_path)

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
