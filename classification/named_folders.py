import os
from scipy.misc import imread
from random import shuffle
import numpy as np
import json
from datasets.tfrecords import PHASE_TRAIN, PHASE_VALIDATION


def one_hot(idx, max_idx):
    label = np.zeros(max_idx, dtype=np.uint8)
    label[idx] = 1
    return label


def crop_center(img,cropy,cropx):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def named_folders(base_dir, phase, prepare_features=None, class_idx={}, crop_roi=None, file_extension=".png", overwrite_cache=False, no_split_folder=False):
    if no_split_folder:
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

    n_classes = len(classes)

    def gen(skip_n=1, offset=0, infinite=False):
        loop_condition = True
        while loop_condition:
            for idx in range(offset, len(images), skip_n):
                if no_split_folder and idx % no_split_folder == 0 and phase == PHASE_TRAIN:
                    continue

                if no_split_folder and idx % no_split_folder != 0 and phase == PHASE_VALIDATION:
                    continue

                feature = imread(images[idx], mode="RGB")
                if crop_roi is not None:
                    feature = crop_center(feature, crop_roi[0], crop_roi[1])
                if prepare_features:
                    feature = prepare_features(feature)
                yield (feature, one_hot(labels[idx], n_classes))
            loop_condition = infinite

    return gen


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Loading Dataset:")
    roi = (200, 200)
    train_data = named_folders("data/lfw-deepfunneled", phase=None, crop_roi=roi, file_extension=".jpg")()

    img, label = next(train_data)
    print("Image shape:")
    print(img.shape)

    for img, label in train_data:
        plt.imshow(img)
        plt.show()
