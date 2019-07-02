import os
import glob
from subprocess import call
from multiprocessing import Pool

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Extract features from wav files.")
parser.add_argument('-i', dest='src', help="Input directory")
parser.add_argument('-o', dest='tgt', help='Output directory')
parser.add_argument('--verbose', dest='verbose')
parser.add_argument('--num_workers', type=int, help='Number of workers', default=4)
parser.add_argument('-C', dest='config_path')

args = parser.parse_args()

os.makedirs(args.tgt, exist_ok=True)
audio_paths = list(glob.glob(os.path.join(args.src, "*")))
print(len(audio_paths))


def process(audio_path):
    file_name = os.path.basename(audio_path)
    htk_path = os.path.join(args.tgt, os.path.splitext(file_name)[0] + '.htk')
    if os.path.exists(htk_path) and os.path.getsize(htk_path) > 0:
        return
    call([
        os.path.join(os.getenv('HCOPY_PATH', 'HCopy')), audio_path, htk_path, "-C", args.config_path])
    if args.verbose:
        tqdm.write("Converting %s." % audio_path)


if __name__ == '__main__':
    with Pool(4) as pool:
        list(tqdm(pool.imap(process, audio_paths), total=len(audio_paths)))
    pool.join()