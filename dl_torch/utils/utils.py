"""General utils"""
import os
import sys
import time
import zipfile
import tarfile
import shutil
from six.moves import urllib
import requests

from .logging import set_log_dir, logger

urllib_start_time = 0

def reporthook(count, block_size, total_size):
    global urllib_start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - urllib_start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def maybe_download(filename, work_directory, source_url):
    """Download the data from source url, unless it's already here.
    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.
    Returns:
        Path to resulting file.
    """
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        r = requests.get(source_url, stream=True)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
    return filepath

    global urllib_start_time
    if not os.path.exists(filepath):
        logger.info("Download file at %s to %s", source_url, filename)
        urllib_start_time = time.time()
        urllib.request.urlretrieve(source_url, filepath, reporthook)
    return filepath


def maybe_unzip(filename, work_directory, folder):
    _dir = os.path.join(work_directory, folder)
    if os.path.exists(_dir):
        return

    _, ext = os.path.splitext(filename)
    if ext == '.zip':
        logger.info("Unzip %s", os.path.join(work_directory, filename))
        zip_ref = zipfile.ZipFile(os.path.join(work_directory, filename), 'r')
        zip_ref.extractall(_dir)
        zip_ref.close()
    elif ext == '.lzma':
        logger.info("Unzip %s", os.path.join(work_directory, filename))
        tar = tarfile.open(filename)
        tar.extractall(path=_dir)
        tar.close()
    else:
        raise Exception("File type is not supported (%s)" % ext)

def init_dirs(params):
    os.makedirs(params.log_dir, exist_ok=True)
    shutil.rmtree(params.output_dir, ignore_errors=True)
    os.makedirs(params.output_dir)
    if params.mode == "train":
        set_log_dir(params)
