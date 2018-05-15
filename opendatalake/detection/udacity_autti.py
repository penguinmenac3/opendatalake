import os
from opendatalake.download_helper import download_dataset as _download

URL = "http://bit.ly/udacity-annotations-autti"
DATASET_FOLDER_NAME = "object-dataset"


def download_dataset(dataset_base_dir):
    """
    Download the dataset to the download dataset base dir.
    :param dataset_base_dir: The directory where the root of your datalake is.
    :return:
    """
    _download(URL, os.path.join(dataset_base_dir, "udacity_autti.tar.gz"))
