import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from subprocess import call

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Convert audio file to wav format.")
parser.add_argument('-i', dest='src', help="Input directory", required=True)
parser.add_argument('-o', dest='tgt', help='Output directory', required=True)
# parser.add_argument('--prefix', dest='prefix')
parser.add_argument('--verbose', action="store_true", dest='verbose')
parser.add_argument('--num_workers', type=int, dest='Number of workers', default=4)

args = parser.parse_args()

os.makedirs(args.tgt, exist_ok=True)
f_trash = open(os.devnull, "w")
audio_paths = list(Path(args.src).glob("**/*.*"))


def process(audio_path: Path):
    if audio_path.suffix not in [".mp3", ".flac"]:
        return
    output_path = Path(args.tgt, audio_path.name)
    output_path = Path(str(output_path).replace(audio_path.suffix, ".wav"))
    if not output_path.exists():
        Path(output_path.parent).mkdir(parents=True, exist_ok=True)
        call(
            ["ffmpeg", "-n", "-i", audio_path, "-ar", "16000", "-ac", "1", output_path],
            stdout=f_trash, stderr=f_trash)
        if args.verbose:
            tqdm.write("Converting %s." % audio_path)


if __name__ == '__main__':
    with Pool(4) as pool:
        list(tqdm(pool.imap(process, audio_paths), total=len(audio_paths)))
    pool.join()