"""General utils"""
import os
import pickle
import re
import subprocess
import sys
import tarfile
import time
import zipfile
from subprocess import call
from typing import List, Union

import requests
from tqdm import tqdm

from .logging import logger

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


def maybe_download(
        download_dir: str,
        source_url: str,
        filename: str = None,
        username: str = None,
        password: str = None) -> str:
    """Download the data from source url, unless it's already here.
    Returns:
        Path to resulting file.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    filepath = os.path.join(download_dir, filename or source_url[source_url.rfind("/")+1:])
    if os.path.exists(filepath):
        return filepath

    with open(filepath, 'wb') as f:
        logger.info("Downloading file at %s to %s", source_url, filepath)

        re_gdrive = r"^https://drive.google.com/[\w]*\?id=([\w\d]*)$"
        if re.match(re_gdrive, source_url):
            m = re.match(re_gdrive, source_url)
            file_id = m[1]
            r = requests.get(
                "https://drive.google.com/uc?export=download", params={'id': file_id},
                stream=True, allow_redirects=True)
        else:
            r = requests.get(source_url, stream=True, allow_redirects=True, auth=(username, password))

        total_length = r.headers.get('content-length')
        if total_length is None:  # no content length header
            for data in r.iter_content(chunk_size=128):
                f.write(data)
        elif r.status_code == 200:
            total_length = int(total_length)
            logger.info("File size: %.1fMB", total_length / 1024 / 1024)

            with tqdm(desc="Downloading", total=int(total_length), unit="B", unit_scale=True,
                      unit_divisor=1024) as pbar:
                for data in r.iter_content(chunk_size=4096):
                    f.write(data)
                    pbar.update(len(data))
    return filepath


def maybe_unzip(file_path, folder_path):
    _dir = folder_path
    if os.path.exists(_dir):
        return

    _, ext = os.path.splitext(file_path)
    try:
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
        elif ext in ['.json']:
            pass
        elif ext == '.rar':
            os.makedirs(folder_path, exist_ok=True)
            process = subprocess.Popen(['unrar', 'x', file_path, folder_path])
            process.communicate()
        else:
            logger.warning("File type is not supported (%s). Not a zip file?" % ext)
    except Exception as e:
        raise Exception(f"Error unzipping {file_path}: {str(e)}")


def load_pkl(filepath: str):
    if os.path.exists(filepath):
        logger.info("Loading from %s...", filepath)
        return pickle.load(open(filepath, "rb"))
    else:
        return None


def dump_pkl(obj, filepath: str):
    logger.info("Saving to %s...", filepath)
    pickle.dump(obj, open(filepath, "wb"))


def camel2snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def run_script(name: str, args):
    import inspect
    import dlex
    root = os.path.dirname(inspect.getfile(dlex))
    call(["python", os.path.join(root, "scripts", name), *args])


def table2str(table: List[List[str]], padding: int = 1) -> str:
    """
    Convert 2D list to table in markdown format
    :param table: list of lists
    :param padding: left and right padding for each cell
    :return: string containing the table
    """
    def _append_blank(s, length):
        return " " * padding + str(s) + " " * (length - len(str(s)) - padding)

    num_cols = len(table[0])
    col_sizes = [max([len(str(row[i])) for row in table]) + 2 * padding for i in range(num_cols)]

    # table header
    s = "|" + "|".join([_append_blank(table[0][i], col_sizes[i]) for i in range(num_cols)]) + "|\n"
    s += "|" + "|".join(["-" * col_sizes[i] for i in range(num_cols)]) + "|\n"

    # table content
    for row in table[1:]:
        s += "|" + "|".join([_append_blank(cell, col_sizes[i]) for i, cell in enumerate(row)]) + "|\n"
    return s


def get_unused_gpus(args):
    try:
        logger.info("Searching for unused GPUs...")
        import subprocess
        from io import StringIO
        import pandas as pd

        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        gpu_df = pd.read_csv(StringIO(gpu_stats.decode()), names=['memory.used', 'memory.free'], skiprows=1)
        gpu_df['memory.used'] = gpu_df['memory.used'].map(lambda x: int(x.rstrip(' [MiB]')))
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
        gpu_df = gpu_df[gpu_df['memory.used'] <= args.gpu_memory_max]
        gpu_df = gpu_df[gpu_df['memory.free'] >= args.gpu_memory_min]
        idx = gpu_df.index.tolist()[:args.num_gpus]
        logger.info("List of GPU(s): %s" % str(idx))
        return idx
    except Exception:
        return []


def get_file_size(filepath: str) -> float:
    return os.path.getsize(filepath) / 1024 / 1024


def split_ints(s: Union[str, int]) -> List[int]:
    return [int(n.strip()) for n in str(s).split(',')]


def prompt(msg: str, default=True) -> bool:
    """
    Prompt yes/no question
    :param msg:
    :param default:
    :return:
    """
    options = "[Y/n]" if default else "[y/N]"
    inp = input(f"{msg} {options}")
    if inp.lower() == 'y':
        return True
    elif inp.lower() == 'n':
        return False
    else:
        return default


def check_interval_passed(last_done: float, interval: str, progress: int = None) -> (bool, float):
    unit = interval[-1]
    value = float(interval[:-1])
    if unit == "e":  # epoch progress (percentage)
        assert progress
        if progress - last_done >= value:
            return True, progress
        else:
            return False, last_done
    elif unit in ["s", "m", "h"]:
        d = dict(s=1, m=60, h=3600)[unit]
        if time.time() / d - last_done > value:
            return True, time.time() / d
        else:
            return False, last_done


def get_num_seconds_from_interval(interval: str) -> Union[int, None]:
    unit = interval[-1]
    value = float(interval[:-1])
    if unit in ["s", "m", "h"]:
        d = dict(s=1, m=60, h=3600)[unit]
        return int(d * value)
    else:
        return None


def get_num_iters_from_interval(interval: str) -> Union[int, None]:
    unit = interval[-1]
    value = float(interval[:-1])
    if unit == "i":
        return int(value)
    else:
        return None


def set_seed(seed):
    import random
    random.seed(seed)
    import numpy
    numpy.random.seed(seed)
    logger.info("Random seed reset to %d", seed)