"""NLP Dataset"""

import os
from struct import unpack
from subprocess import call

import nltk
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from dlex.datasets.base.nlp import Vocab, get_token_id, load_tkn_to_idx
from dlex.utils.logging import logger, beautify
from dlex.utils.ops_utils import FloatTensor, LongTensor
from .base import BaseDataset


def extract_features(processed_dir, file_paths):
    """
    Extract features and calculate mean and var
    :type processed_dir: str
    :param dict[str, list[str]] file_paths: {'train': list, 'test': list} filenames without extension
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
        working_dir = os.path.join(processed_dir, mode)
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
                        call(["ffmpeg", "--threads", 10, "-i", file_path, wav_filepath], stdout=f_trash, stderr=f_trash)
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
    np.save(os.path.join(processed_dir, "mean.npy"), mean)
    np.save(os.path.join(processed_dir, "var.npy"), var)


def regularize(processed_dir, file_paths, transcripts):
    """

    :param str processed_dir:
    :param [str] file_paths:
    :param [str] transcripts:
    """
    logger.info("Write outputs to file")
    mean = np.load(os.path.join(processed_dir, "mean.npy"))
    var = np.load(os.path.join(processed_dir, "var.npy"))

    for mode in ["test", "train"]:
        os.makedirs(os.path.join(processed_dir, mode, "npy"), exist_ok=True)
        for file_path, transcript in tqdm(list(zip(file_paths[mode], transcripts[mode])), desc=mode):
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


def write_dataset(processed_dir, output_prefix, file_paths, transcripts, vocab_path, normalize_fn, tokenize_fn):
    """
    :param str processed_dir:
    :param str output_prefix: Output file is saved to {output_prefix}_{train/test}.csv
    :param dict[str, list[str]] file_paths:
    :param dict[str, list[str]] transcripts:
    :param str vocab_path:
    :param (str) -> str normalize_fn:
    :param (str) -> list[str] tokenize_fn:
    """
    outputs = {'train': [], 'test': []}
    vocab = load_tkn_to_idx(vocab_path)
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
                    target=' '.join([str(get_token_id(vocab, tkn)) for tkn in tokenize_fn(normalize_fn(transcript))]),
                    trans_words=transcript
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


class VoiceBaseDataset(BaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)


class SpeechRecognitionBaseDataset(VoiceBaseDataset):
    def __init__(self, mode, params, vocab_path):
        super().__init__(mode, params)
        self._vocab = Vocab(vocab_path)
        self._vocab.add_token('<sos>')
        self._vocab.add_token('<eos>')
        self._output_size = len(self._vocab)

    @property
    def output_size(self):
        """
        :rtype: int
        """
        return self._output_size

    @property
    def sos_token_idx(self):
        return self._vocab.sos_token_idx

    @property
    def eos_token_id(self):
        return self._vocab.eos_token_idx

    def collate_fn(self, batch):
        for item in batch:
            item['X'] = np.load(item['X_path'])

        batch.sort(key=lambda item: len(item['X']), reverse=True)
        inp = [FloatTensor(item['X']) for item in batch]
        tgt = [LongTensor(item['Y']).view(-1) for item in batch]
        inp = nn.utils.rnn.pad_sequence(
            inp, batch_first=True)
        tgt = nn.utils.rnn.pad_sequence(
            tgt, batch_first=True,
            padding_value=self.eos_token_id)

        return dict(
            X_path=item['X_path'],
            X=inp, X_len=LongTensor([len(item['X']) for item in batch]),
            Y=tgt, Y_len=LongTensor([len(item['Y']) for item in batch]))

    def evaluate(self, y_pred, batch, metric):
        dist, count = 0, 0
        for pr, gt, gt_len in zip(y_pred, batch['Y'], batch['Y_len']):
            pr = np.array(pr)[:-1]
            gt = gt.cpu().detach().numpy()
            gt = gt[1:gt_len - 1]
            dist += nltk.edit_distance(pr, gt)
            count += len(gt)
        return dist, count

    def format_output(self, y_pred, batch_input):
        pr = np.array(y_pred)[:-1]
        gt = batch_input['Y'].cpu().detach().numpy()
        gt = gt[1:batch_input['Y_len'] - 1]
        if self.params.dataset.output_format is None:
            return "", str(gt), str(pr)
        elif self.params.dataset.output_format == "text":
            delimiter = ' ' if self.params.dataset.unit == "word" else ''
            return \
                batch_input['X_path'], \
                delimiter.join([self._vocab.get_token(wid) for wid in gt]), \
                delimiter.join([self._vocab.get_token(wid) for wid in pr])
