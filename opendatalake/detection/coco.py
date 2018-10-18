from __future__ import print_function

import json
import time
import os
import sys
import math
from scipy.misc import imread
import numpy as np
from PIL import Image
from zipfile import ZipFile

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

import tensorflow as tf
from opendatalake.detection.utils import Detection25d, Detection2d, apply_projection, vec_len

Sequence = tf.keras.utils.Sequence
PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"

COCO2014_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO2017_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

DATASETS = ["val2014", "val2017", "train2014", "train2017"]


class COCO(Sequence):
    def __init__(self, hyperparams, phase, preprocess_feature=None, preprocess_label=None, augment_data=None):
        self.base_dir = hyperparams.problem.get("data_path", "datasets/coco")
        self.batch_size = hyperparams.train.get("batch_size", 16)
        self.phase = phase
        self.hyperparams = hyperparams
        self.preprocess_feature = preprocess_feature
        self.preprocess_label = preprocess_label
        self.augment_data = augment_data

        annotation_file = '{}/annotations/instances_{}.json'.format(self.base_dir, phase)

        # If data does not exist download it.
        if not os.path.exists(annotation_file):
            if not os.path.exists(self.base_dir + "/trainval2014.zip"):
                print("Download: " + COCO2014_ANNOTATIONS_URL)
                urlretrieve(COCO2014_ANNOTATIONS_URL, self.base_dir + "/trainval2014.zip")
                print("Extracting: {}/trainval2014.zip".format(self.base_dir))
                zip_ref = ZipFile(self.base_dir + "/trainval2014.zip", 'r')
                zip_ref.extractall(self.base_dir)
                zip_ref.close()
            if not os.path.exists(self.base_dir + "/trainval2017.zip"):
                print("Download: " + COCO2017_ANNOTATIONS_URL)
                urlretrieve(COCO2017_ANNOTATIONS_URL, self.base_dir + "/trainval2017.zip")
                print("Extracting: {}/trainval2017.zip".format(self.base_dir))
                zip_ref = ZipFile(self.base_dir + "/trainval2017.zip", 'r')
                zip_ref.extractall(self.base_dir)
                zip_ref.close()

            for dataset in DATASETS:
                print("Download: " + dataset)
                self._download(data_type=dataset, data_dir=self.base_dir)

        self.dataset = json.load(open(annotation_file, 'r'))
        self.N = len(self.dataset['images'])

        self.annotations = {}
        for annotation in self.dataset["annotations"]:
            if annotation["image_id"] not in self.annotations:
                self.annotations[annotation["image_id"]] = []
            self.annotations[annotation["image_id"]].append(annotation)

    def categories(self):
        return self.dataset["categories"]

    def category_map(self):
        category_map = {}
        for category in self.categories():
            category_map[category["id"]] = category["name"]
        return category_map

    def __len__(self):
        return math.floor(self.N / self.batch_size)

    def __getitem__(self, index):
        features = []
        labels = []
        for idx in range(index * self.batch_size, min((index + 1) * self.batch_size, self.N)):
            image_id = self.dataset['images'][idx]['id']
            filename = '%s/images/%s/%s' % (self.base_dir, self.phase, self.dataset['images'][idx]['file_name'])

            detections2d = []
            instance_masks = []
            if image_id in self.annotations:
                for annotation in self.annotations[image_id]:
                    class_id = annotation["category_id"]
                    w = annotation["bbox"][2]
                    h = annotation["bbox"][3]
                    cx = annotation["bbox"][0] + w / 2
                    cy = annotation["bbox"][1] + h / 2
                    mask = list(annotation["segmentation"])
                    detections2d.append(Detection2d(class_id, cx, cy, w, h, conf=1.0, instance_mask=mask))

            feature = imread(filename, mode="RGB")
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

    def _download(self, data_type, data_dir="data/coco"):
        annotation_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        tar_dir = '%s/images/%s' % (data_dir, data_type)
        print("Loading annotation file: " + annotation_file)
        dataset = json.load(open(annotation_file, 'r'))

        imgs = {}
        for img in dataset['images']:
                    imgs[img['id']] = img

        N = len(imgs)
        if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)

        print("Number of images: " + str(N))
        i = 0
        for img in imgs.values():
            err = True
            while err:
                tic = time.time()
                try:
                    filename = '%s/images/%s/%s' % (data_dir, data_type, img['file_name'])
                    if not os.path.exists(filename):
                        urlretrieve(img['coco_url'], filename)
                    err = False
                except IOError:
                    print("IOError retrying.")
                    continue
                print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time()- tic), end="\r")
            i += 1

    def verify_integrity(self, auto_repair=False):
        N = len(self.dataset["images"])
        errorous_images = []
        for idx in range(N):
            filename = '%s/images/%s/%s' % (self.base_dir, self.phase, self.dataset['images'][idx]['file_name'])
            err = False
            try:
                image = np.asarray(Image.open(filename).convert('RGB'))
                image = image[:, :, ::-1]
            except IOError:
                errorous_images.append(filename)
                print("Found broken image: {}".format(filename))
                err = True
            while err and auto_repair:
                tic = time.time()
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                    urlretrieve(self.dataset['images'][idx]['coco_url'], filename)
                    err = False
                except IOError:
                    print("IOError retrying.")
                    continue
                print('Repaired image {} (t={:0.1f}s)'.format(idx, time.time() - tic))
            print('Checked {}/{} images'.format(idx, N), end="\r")
        return errorous_images
