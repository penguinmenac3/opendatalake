## OpenDataLake [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Since it is all about data, this are data wrappers for common datasets which load the data into a common structure.

There are handlers for several datasets.
To get you started quickly.

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

1. Coco [TODO]
2. CamVid [TODO]
3. Cityscapes [TODO]


## Regression

Here are all regression datasets.
They are loaded as a generator spitting out a feature(-vector/image) regression(-vector) the size of the vector to regress depends on the problem.

### 2d-Detection (bounding boxes)

The data structure is difficult to describe, but this sample should make it quite obvious:
```python
feature, regression = next(data)
for instance in regression:
    one_hot_class_label = instance[:num_classes]
    x, y, width, height = instance[num_classes:]
```

To make stuff simple there is a visualize-method.
```python
image = visualize(image, x, y, width, height, class_label=None)
```

1. Kitti 2d [TODO]
2. Tensorbox [TODO]
3. ImageNet Object Localization [TODO]

### 3d-Detection (bounding boxes)

The data structure similar to 2d detection and difficult to describe, but this sample should make it quite obvious:

```python
feature, regression = next(data)
for instance in regression:
    one_hot_class_label = instance[:num_classes]
    x, y, width, height = instance[num_classes:num_classes + 4]
    x3d, y3d, z3d, width3d, heigh3d, length3d, roll, pitch, yaw = instance[num_classes + 4:]
```

As stuff gets complicated here, there are helper functions like project returning a list of 2d points.
As well as a visualize method.
Because debugging is hell in 3d.

```python
[(x1, y1), ..., (x8, y8)] = project(x, y, z, width, heigh, length, roll, pitch, yaw)
image = visualize(image, [(x1, y1), ..., (x8, y8)])
```

1. Kitti 3d [TODO]

Are there more? Please let me know.

### Misc

Nothing here yet.
However, there might be regression datasets that I will put here.
