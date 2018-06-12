"""
This package implements texture augmentation as described in "Train Here, Deploy There: Robust Segmentation in Unseen Domains"
"""

import random
import numpy as np
import cv2


def full_texture_augmentation(image):
    image = image.copy()
    image = saturation(image).astype(np.float64)
    image = contrast(image)
    image = brightness(image)
    image = jitter(image)
    #image = salt(image)
    #image = pepper(image)
    image = value_crop(image)

    return image.astype(np.uint8)


def value_crop(image, upper_bound=255, lower_bound=0):
    image[image > upper_bound] = upper_bound
    image[image < lower_bound] = lower_bound
    return image


def brightness(image, upper_bound=(45, 45, 45), lower_bound=(-45, -45, -45), random_1=random.random()):
    for i in range(image.shape[2]):
        image[:, :, i] += (random_1 * upper_bound[i] + (1-random_1) * lower_bound[i])
    return image


def contrast(image, upper_bound=(1.25, 1.25, 1.25), lower_bound=(0.75, 0.75, 0.75), random_1=random.random()):
    image[:, :, :] -= 128.0
    for i in range(image.shape[2]):
        image[:, :, i] *= random_1 * upper_bound[i] + (1-random_1) * lower_bound[i]
    image[:, :, :] += 128.0
    return image


def saturation(image, upper_bound=1.5, lower_bound=0.25, random_1=random.random()):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = image.astype(np.float64)
    image[:, :, 1] *= random_1 * upper_bound + (1-random_1) * lower_bound
    image = value_crop(image).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def jitter(image, upper_bound=13, lower_bound=-13):
    noise = np.random.randint(low=lower_bound, high=upper_bound, size=image.shape)
    tmp = image + noise
    return tmp


def salt(image, prob=0.05):
    h, w, c = image.shape
    noise = np.random.random(size=(h, w))
    image[noise < prob, :] = 255
    return image


def pepper(image, prob=0.05):
    h, w, c = image.shape
    noise = np.random.random(size=(h, w))
    image[noise < prob, :] = 0
    return image
