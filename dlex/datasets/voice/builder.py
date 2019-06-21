"""NLP Dataset"""

import os
from struct import unpack
from subprocess import call
from typing import List, Dict, Callable

import nltk
import numpy as np
from tqdm import tqdm

from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.builder import DatasetBuilder
from dlex.utils.logging import logger, beautify


class VoiceDatasetBuilder(DatasetBuilder):
    def extract_features(
            self,
            file_paths: Dict[str, List[str]]):
        """
        Extract features and calculate mean and var
        :param processed_dir:
        :param file_paths: {'train': list, 'test': list} filenames without extension
        """
        # if os.path.exists(os.path.join(processed_dir, "mean.npy")):
        #     return
        logger.info("Extracting features...")
        # get mean
        mean = np.array([0] * 120)
        var = np.array([0] * 120)
        count = 0

        f_trash = open(os.devnull, "w")
        for mode in ["train", "test"]:
            working_dir = os.path.join(self.get_processed_data_dir(), mode)
            os.makedirs(os.path.join(working_dir, "waves"), exist_ok=True)
            os.makedirs(os.path.join(working_dir, "features"), exist_ok=True)

            for file_path in tqdm(file_paths[mode], desc=mode):
                file_name = os.path.basename(file_path)
                try:
                    file_name, file_ext = os.path.splitext(file_name)

                    # convert to wav
                    if file_ext == ".mp3":
                        wav_filepath = os.path.join(working_dir, "waves", file_name + '.wav')
                        if not os.path.exists(wav_filepath):
                            call(
                                ["ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1", wav_filepath],
                                stdout=f_trash,
                                stderr=f_trash)
                    elif file_ext == ".wav":
                        wav_filepath = file_path
                    else:
                        raise Exception("Unsupported file type %s" % file_ext)

                    # export feature
                    htk_filename = os.path.join(working_dir, "features", file_name + ".htk")
                    if not os.path.exists(htk_filename):
                        call([
                            os.path.join(os.getenv('HCOPY_PATH', 'HCopy')),
                            wav_filepath,
                            htk_filename,
                            "-C", "config.lmfb.40ch"
                        ])

                    # update mean and var
                    if mode == "train":
                        fh = open(htk_filename, "rb")
                        spam = fh.read(12)
                        nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
                        veclen = int(sampSize / 4)
                        fh.seek(12, 0)
                        dat = np.fromfile(fh, dtype=np.float32)
                        dat = dat.reshape(len(dat) // veclen, veclen)
                        dat = dat.byteswap()
                        fh.close()

                        for k in range(len(dat)):
                            updated_mean = (mean * count + dat[k]) / (count + 1)
                            var = (count * var + (dat[k] - mean) * (dat[k] - updated_mean)) / (count + 1)
                            mean = updated_mean
                            count += 1
                except Exception as e:
                    logger.error("Error processing %s (%s)", file_path, str(e))
        f_trash.close()
        logger.debug("mean: %s", beautify(mean))
        logger.debug("var: %s", beautify(var))
        np.save(os.path.join(self.get_processed_data_dir(), "mean.npy"), mean)
        np.save(os.path.join(self.get_processed_data_dir(), "var.npy"), var)

    def regularize(
            self,
            file_paths: Dict[str, List[str]]):
        processed_dir = self.get_processed_data_dir()
        logger.info("Write outputs to file")
        mean = np.load(os.path.join(processed_dir, "mean.npy"))
        var = np.load(os.path.join(processed_dir, "var.npy"))

        for mode in ["test", "train"]:
            os.makedirs(os.path.join(processed_dir, mode, "npy"), exist_ok=True)
            for file_path in tqdm(list(file_paths[mode]), desc=mode):
                file_name = os.path.basename(file_path)
                try:
                    file_name, _ = os.path.splitext(file_name)
                    if file_name == "":
                        continue
                    npy_filename = os.path.join(processed_dir, mode, "npy", file_name + ".npy")

                    if True:
                        # (rate, sig) = wav.read(wav_filename)
                        htk_filename = os.path.join(processed_dir, mode, "features", file_name + ".htk")
                        fh = open(htk_filename, "rb")
                        spam = fh.read(12)
                        nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
                        veclen = int(sampSize / 4)
                        fh.seek(12, 0)
                        dat = np.fromfile(fh, dtype=np.float32)
                        dat = dat.reshape(len(dat) // veclen, veclen)
                        dat = dat.byteswap()
                        fh.close()

                        dat = (dat - mean) / np.sqrt(var)
                        np.save(npy_filename, dat)
                except Exception as e:
                    logger.error("Error processing %s (%s)", file_path, str(e))

    def write_dataset(
            self,
            output_prefix: str,
            file_paths: Dict[str, List[str]],
            transcripts: Dict[str, List[str]],
            vocab_path: str,
            normalize_fn: Callable[[str], str],
            tokenize_fn: Callable[[str], List[str]]):
        processed_dir = self.get_processed_data_dir()
        print(processed_dir)
        outputs = {'train': [], 'test': []}
        vocab = Vocab(vocab_path)
        for mode in ["test", "train"]:
            os.makedirs(os.path.join(processed_dir, mode, "npy"), exist_ok=True)
            for file_path, transcript in tqdm(list(zip(file_paths[mode], transcripts[mode])), desc=mode):
                file_name = os.path.basename(file_path)
                file_name, _ = os.path.splitext(file_name)
                if file_name == "":
                    continue
                npy_filename = os.path.join(processed_dir, mode, "npy", file_name + ".npy")
                if os.path.exists(npy_filename):
                    outputs[mode].append(dict(
                        filename=npy_filename,
                        target=' '.join(
                            [str(vocab[tkn]) for tkn in tokenize_fn(normalize_fn(transcript))]),
                        trans_words=normalize_fn(transcript)
                    ))

        for mode in ["test", "train"]:
            # outputs[mode].sort(key=lambda item: len(item['target_word']))
            fn = os.path.join(processed_dir, "%s_%s" % (output_prefix, mode) + '.csv')
            logger.info("Output to %s" % fn)
            with open(fn, 'w', encoding='utf-8') as f:
                f.write('\t'.join(['sound', 'target', 'trans']) + '\n')
                for o in outputs[mode]:
                    f.write('\t'.join([
                        o['filename'],
                        o['target'],
                        o['trans_words']
                    ]) + '\n')

    def evaluate(self, pred, ref, metric: str):
        if metric == "wer":
            return nltk.edit_distance(pred, ref), len(ref)