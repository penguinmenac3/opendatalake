## OpenDataLake [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Since it is all about data, this are data wrappers for common datasets which load the data into a common structure.

There are handlers for several datasets.
To get you started quickly.

## Install

Simply install it via pip.

```bash
pip install opendatalake
```

## Classification

Here are all classification datasets.
They are loaded as a generator spitting out a feature(-vector/image) and a one-hot-encoded label.

1. [Named Folders (Foldername = Label)](opendatalake/classification/named_folders.py)
2. [MNIST](opendatalake/classification/mnist.py)
3. ImageNet [TODO]
4. [Cifar10/Cifar100](opendatalake/classification/cifar.py)
5. [LFW (named folders)](opendatalake/classification/named_folders.py)
6. PASCAL VOC [TODO]
7. Places [TODO]

## Segmentation

Here are all segmentation datasets.
They are loaded as a generator spitting out a feature(-vector/image) and segmentation(-vector/image).

1. [Coco (WIP)](opendatalake/segmentation/coco.py)
2. CamVid [TODO]
3. Cityscapes [TODO]


## Detection

Here are all detection datasets.
They are loaded as a generator spitting out a feature(-vector/image) and a detections(-vector/image).
A detection is either a detection_2d, detection_2.5d or detection_3d object defined [here](opendatalake/detection/utils.py).

1. [Bosch TLR](opendatalake/detection/bosch_tlr.py)
2. [Coco (WIP)](opendatalake/detection/coco.py)
3. [Kitti Detection (incl. 3d)](opendatalake/detection/kitti_detection.py)
4. [Pascal Voc 3d (WIP)](opendatalake/detection/pascal_voc_3d.py)
5. [Udacity Autti (WIP)](opendatalake/detection/udacity_autti.py)
6. [Udacity Crowdai (WIP)](opendatalake/detection/udacity_crowdai.py)

## Unlabeled

Here are all unlabeled datasets.
They are loaded as a generator spitting out a feature(-vector/image).

1. [Image Folder](opendatalake/unlabeled/image_folder.py)
