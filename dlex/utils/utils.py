"""General utils"""
import os
import sys
import time
import zipfile
import tarfile
import shutil
import re
from subprocess import call

from six.moves import urllib
import requests
from tqdm import tqdm

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


def maybe_download(download_dir: str, source_url: str, filename: str = None) -> str:
    """Download the data from source url, unless it's already here.
    Returns:
        Path to resulting file.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    filepath = os.path.join(download_dir, filename or source_url[source_url.rfind("/")+1:])
    if not os.path.exists(filepath):
        with open(filepath, 'wb') as f:
            logger.info("Downloading file at %s to %s", source_url, filepath)
            r = requests.get(source_url, stream=True, allow_redirects=True)

            total_length = r.headers.get('content-length')
            if total_length is None:  # no content length header
                for data in r.iter_content(chunk_size=128):
                    f.write(data)
                    print(len(data))
            elif r.status_code == 200:
                total_length = int(total_length)
                logger.info("File size: %.1fMB", total_length / 1024 / 1024)

                with tqdm(desc="Downloading", total=int(total_length), unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                    for data in r.iter_content(chunk_size=4096):
                        f.write(data)
                        pbar.update(len(data))
    return filepath


def maybe_unzip(file_path, folder_path):
    _dir = folder_path
    if os.path.exists(_dir):
        return

    _, ext = os.path.splitext(file_path)
    if ext == '.zip':
        logger.info("Extract %s to %s", file_path, folder_path)
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(_dir)
        zip_ref.close()
    elif ext in ['.lzma', '.gz', '.tgz']:
        logger.info("Extract %s to %s", file_path, folder_path)
        tar = tarfile.open(file_path)
        tar.extractall(path=_dir)
        tar.close()
    else:
        raise Exception("File type is not supported (%s)" % ext)


def init_dirs(params):
    os.makedirs(params.log_dir, exist_ok=True)
    # shutil.rmtree(params.output_dir, ignore_errors=True)
    os.makedirs(params.output_dir, exist_ok=True)
    if params.mode == "train":
        set_log_dir(params)


def camel2snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def run_script(name: str, args):
    import inspect
    import dlex
    root = os.path.dirname(inspect.getfile(dlex))
    call(["python", os.path.join(root, "scripts", name), *args])