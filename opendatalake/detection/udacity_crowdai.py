import os
from opendatalake.download_helper import download_dataset as _download

URL = "http://bit.ly/udacity-annoations-crowdai"
DATASET_FOLDER_NAME = "object-detection-crowdai"


def download_dataset(dataset_base_dir):
    """
    Download the dataset to the download dataset base dir.
    :param dataset_base_dir: The directory where the root of your datalake is.
    :return:
    """
    _download(URL, os.path.join(dataset_base_dir, "udacity_crowdai.tar.gz"))
