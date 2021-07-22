"""Prepare PASCAL VOC datasets
Modified: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/utils/files.py
"""
import argparse
import errno
import hashlib
import os
import shutil
import tarfile

import requests
from tqdm import tqdm


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split("/")[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path

    if (
        overwrite
        or not os.path.exists(fname)
        or (sha1_hash and not check_sha1(fname, sha1_hash))
    ):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print("Downloading %s from %s..." % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get("content-length")
        with open(fname, "wb") as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=int(total_length / 1024.0 + 0.5),
                    unit="KB",
                    unit_scale=False,
                    dynamic_ncols=True,
                ):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning(
                f"File {fname} is downloaded but the content hash does not match. "
                "The repo may be outdated or download may be incomplete. "
                'If the "repo_url" is overridden, consider switching to '
                "the default repo."
            )

    return fname


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def mkdir(path):
    """make dir exists okay"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def download_voc(path, overwrite=False):
    _DOWNLOAD_URLS = [
        (
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "4e443f8a2eca6b1dac8a6c57641b67dd40621a49",
        )
    ]
    download_dir = os.path.join(path, "downloads")
    mkdir(download_dir)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(
            url, path=download_dir, overwrite=overwrite, sha1_hash=checksum
        )
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)


def download_aug(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        (
            "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
            "7129e0a480c2d6afb02b517bb18ac54283bfaa35",
        )
    ]
    download_dir = os.path.join(path, "downloads")
    mkdir(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(
            url, path=download_dir, overwrite=overwrite, sha1_hash=checksum
        )
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)
            shutil.move(
                os.path.join(path, "benchmark_RELEASE"),
                os.path.join(path, "VOCaug"),
            )
            filenames = ["VOCaug/dataset/train.txt", "VOCaug/dataset/val.txt"]
            # generate trainval.txt
            with open(
                os.path.join(path, "VOCaug/dataset/trainval.txt"), "w"
            ) as outfile:
                for fname in filenames:
                    fname = os.path.join(path, fname)
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)


if __name__ == "__main__":
    _TARGET_DIR = os.path.expanduser("./datasets")
    mkdir(_TARGET_DIR)
    download_voc(_TARGET_DIR, overwrite=False)
    download_aug(_TARGET_DIR, overwrite=False)
