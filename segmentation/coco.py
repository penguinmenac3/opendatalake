import json
import time
import os
import sys
from scipy.misc import imread
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


DATASETS = ["val2014", "val2017", "train2014", "train2017"]


def download(data_type, data_dir="data/coco"):
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
        tic = time.time()
        filename = '%s/images/%s/%s' % (data_dir, data_type, img['file_name'])
        if not os.path.exists(filename):
            urlretrieve(img['coco_url'], filename)
        print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time()- tic))
        i += 1


def download_all(data_dir="data/coco"):
    for dataset in DATASETS:
        print("Download: " + dataset)
        download(data_type=dataset, data_dir=data_dir)


def _gen(params, skip_n=1, offset=0):
    base_dir, phase, dataset, N, prepare_features, prepare_labels = params

    for idx in range(offset, N, skip_n):
        filename = '%s/images/%s/%s' % (base_dir, phase, dataset['images'][idx]['file_name'])
        feature = imread(filename, mode="RGB")
        labels = None  # TODO load instance segmentation and bounding boxes
        if prepare_features:
            feature = prepare_features(feature)
        if prepare_labels:
            labels = prepare_labels(labels)
        yield (feature, labels)


def coco(base_dir, phase, prepare_features=None, prepare_labels=None):
    annotation_file = '{}/annotations/instances_{}.json'.format(base_dir, phase)
    print("Loading annotation file: " + annotation_file)
    dataset = json.load(open(annotation_file, 'r'))

    N = len(dataset['images'])
    print("Number of images: " + str(N))

    return _gen, (base_dir, phase, dataset, N, prepare_features, prepare_labels)


if __name__ == "__main__":
    download_all()
