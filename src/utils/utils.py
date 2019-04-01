import os, sys
from six.moves import urllib
import time
import zipfile

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
    global urllib_start_time
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print("Download file from", source_url)
        urllib_start_time = time.time()
        urllib.request.urlretrieve(source_url, filepath, reporthook)
    return filepath


def maybe_unzip(filename, work_directory, folder):
    if not os.path.exists(os.path.join(work_directory, folder)):
        print("Unzip", os.path.join(work_directory, filename))
        zip_ref = zipfile.ZipFile(os.path.join(work_directory, filename), 'r')
        zip_ref.extractall(os.path.join(work_directory, folder))
        zip_ref.close()
