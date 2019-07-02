import os
import glob
from subprocess import call
from multiprocessing import Pool

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Convert audio file to wav format.")
parser.add_argument('-i', dest='src', help="Input directory")
parser.add_argument('-o', dest='tgt', help='Output directory')
parser.add_argument('--prefix', dest='prefix')
parser.add_argument('--verbose', dest='verbose')
parser.add_argument('--num_workers', type=int, dest='Number of workers', default=4)

args = parser.parse_args()

os.makedirs(args.tgt, exist_ok=True)
f_trash = open(os.devnull, "w")
audio_paths = list(glob.glob(os.path.join(args.src, "*")))


def process(audio_path):
    file_name = os.path.basename(audio_path)
    file_name = os.path.splitext(file_name)[0]
    if args.prefix:
        file_name = "%s_%s" % (args.prefix, file_name)
    wav_path = os.path.join(args.tgt, file_name + '.wav')
    call(
        ["ffmpeg", "-n", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
        stdout=f_trash,
        stderr=f_trash)
    if args.verbose:
        tqdm.write("Converting %s." % audio_path)


if __name__ == '__main__':
    with Pool(4) as pool:
        list(tqdm(pool.imap(process, audio_paths), total=len(audio_paths)))
    pool.join()