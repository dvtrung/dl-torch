import argparse
import glob
import os
from struct import unpack

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Extract features from wav files.")
parser.add_argument('-i', dest='src', help="Input directory")
parser.add_argument('-o', dest='tgt', help='Output directory')
parser.add_argument('--verbose', dest='verbose')
parser.add_argument('--num_workers', type=int, help='Number of workers', default=4)
parser.add_argument('-C', dest='config_path')

args = parser.parse_args()

os.makedirs(args.tgt, exist_ok=True)
htk_paths = list(glob.glob(os.path.join(args.src, "*")))


def read_htk(path):
    fh = open(path, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(len(dat) // veclen, veclen)
    dat = dat.byteswap()
    fh.close()
    return dat


if __name__ == '__main__':
    num_filters = 120
    mean = np.array([0] * num_filters)
    var = np.array([0] * num_filters)
    count = 0
    for htk_path in tqdm(htk_paths):
        if os.path.getsize(htk_path) == 0:
            print(htk_path)
            continue
        try:
            feat = read_htk(htk_path)
            for k in range(len(feat)):
                updated_mean = (mean * count + feat[k]) / (count + 1)
                var = (count * var + (feat[k] - mean) * (feat[k] - updated_mean)) / (count + 1)
                mean = updated_mean
                count += 1
        except:
            print("Error while processing %s" % htk_path)

    np.save(os.path.join(args.tgt, "mean.npy"), mean)
    np.save(os.path.join(args.tgt, "var.npy"), var)