download = None
try:
    import urllib
    download = urllib.urlretrieve
except:
    import urllib.request
    download = urllib.request.urlretrieve


def download_dataset(url, download_path):
    """
    Downloads the file at the specified url to the download path.
    :param url: The download url.
    :param download_path: The path to the file where to store the download.
    :return:
    """
    download(url, download_path)
