from opendatalake.segmentation.coco import download_all, download, DATASETS
from opendatalake.segmentation.coco import coco as _coco


def coco(base_dir, phase, prepare_features=None, prepare_labels=None):
    # TODO prepare labels in a way only bounding boxes are left:
    return _coco(base_dir, phase, prepare_features, prepare_labels)


if __name__ == "__main__":
    download_all()
