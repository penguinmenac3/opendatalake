import os
from scipy.misc import imread
from random import shuffle
import numpy as np
import json


def crop_center(img,cropy,cropx):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def _gen(params, skip_n=1, offset=0, infinite=False):
    images, crop_roi, prepare_features, add_noise = params
    loop_condition = True
    while loop_condition:
        for idx in range(offset, len(images), skip_n):
            feature = imread(images[idx], mode="RGB")
            if crop_roi is not None:
                feature = crop_center(feature, crop_roi[0], crop_roi[1])
            if prepare_features:
                feature = prepare_features(feature)
            noisy_feature = feature
            if add_noise:
                pass  # TODO add noise
            yield (noisy_feature, feature)
        loop_condition = infinite


def image_folder(base_dir, phase, prepare_features=None, crop_roi=None, file_extension=".png", overwrite_cache=False, add_noise=False):
    if phase is not None:
        data_dir = os.path.join(base_dir, phase)
    else:
        data_dir = base_dir
    images = []

    if overwrite_cache:
        if os.path.exists(os.path.join(data_dir, "images.json")):
            os.remove(os.path.join(data_dir, "images.json"))

    if os.path.exists(os.path.join(data_dir, "images.json")):
        print("Using buffer files.")
        with open(os.path.join(data_dir, "images.json"), 'r') as infile:
            images = json.load(infile)
    else:
        print("No buffer files found. Reading folder structure and creating buffer files.")
        for filename in os.listdir(data_dir):
            if filename.endswith(file_extension):
                images.append(os.path.join(data_dir, filename))

        with open(os.path.join(data_dir, "images.json"), 'w') as outfile:
            json.dump(images, outfile)

    return _gen, (images, crop_roi, prepare_features, add_noise)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Loading Dataset:")
    roi = (200, 200)
    train_data = image_folder("data/lfw-deepfunneled", phase=None, crop_roi=roi, file_extension=".jpg")

    data_fn, data_params = train_data
    data_gen = data_fn(data_params)

    noisy_img, img = next(data_gen)
    print("Image shape:")
    print(img.shape)

    for noisy_img, img in data_gen:
        plt.imshow(img)
        plt.show()
