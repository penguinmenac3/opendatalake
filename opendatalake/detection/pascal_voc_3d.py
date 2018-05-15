import os
from opendatalake.download_helper import download_dataset as _download

URL = "ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip"
DATASET_FOLDER_NAME = "PASCAL3D+_release1.1"


def download_dataset(dataset_base_dir):
    """
    Download the dataset to the download dataset base dir.
    :param dataset_base_dir: The directory where the root of your datalake is.
    :return:
    """
    _download(URL, os.path.join(dataset_base_dir, "pascal_voc_3d.zip"))
