import os

import numpy as np
from tqdm import tqdm as tqdm
from subprocess import call
from struct import unpack, pack
import torch.nn as nn
import nltk

from dlex.utils.logging import logger
from dlex.datasets.base.voice import VoiceDataset
from dlex.datasets.base.nlp import get_token_id, load_tkn_to_idx, normalize_word, normalize_char, Vocab
from dlex.utils.ops_utils import LongTensor, FloatTensor

DOWNLOAD_URL = "https://ailab.hcmus.edu.vn/assets/vivos.tar.gz"


def build_vocab(raw_dir, processed_dir):
    mode = "train"

    wset = set()
    with open(os.path.join(raw_dir, mode, "prompts.txt"), encoding="utf-8") as f:
        for s in f.read().split('\n'):
            s = s.replace(':', '')
            words = s.split(' ')[1:]
            for word in words:
                if word != '':
                    wset.add(normalize_word(word))

    # word-unit
    with open(os.path.join(processed_dir, "vocab_words.txt"), "w", encoding="utf-8") as f:
        f.write("<pad>\n<oov>\n")
        f.write("\n".join(["%s" % word for _, word in
                           enumerate(list(wset)) if word != ""]))

    # char-unit
    cset = set('_')
    for word in wset: cset |= set(word)
    with open(os.path.join(processed_dir, "vocab_chars.txt"), "w", encoding="utf-8") as f:
        f.write("<pad>\n<oov>\n")
        f.write("\n".join(["%s" % c for _, c in
                           enumerate(list(cset)) if c != ""]))

    vocab_word = {word: i for i, word in enumerate(
        open(
            os.path.join(processed_dir, "vocab_words.txt"),
            encoding='utf-8').read().split('\n'))}
    vocab_char = {word: i for i, word in enumerate(
        open(
            os.path.join(processed_dir, "vocab_chars.txt"),
            encoding='utf-8').read().split('\n'))}

    logger.info("Word count: %d, Char count: %d", len(vocab_word), len(vocab_char))


def calculate_mean_and_var(raw_dir, processed_dir):
    logger.info("Extract features")
    # get mean
    mean = np.array([0] * 120)
    var = np.array([0] * 120)
    count = 0

    for mode in ["train", "test"]:
        with open(os.path.join(raw_dir, mode, "prompts.txt"), encoding="utf-8") as f:
            lines = f.read().split("\n")
            for i, s in tqdm(list(enumerate(lines)), desc=mode):
                filename = s.split(' ')[0]
                if filename == "":
                    continue
                wav_filename = os.path.join(raw_dir, mode, "waves", filename.split('_')[0], filename + ".wav")

                os.makedirs(os.path.join(processed_dir, mode, "features"), exist_ok=True)
                htk_filename = os.path.join(processed_dir, mode, "features", filename + ".htk")
                call([
                    os.path.join(os.getenv('HCOPY_PATH', 'HCopy')),
                    wav_filename,
                    htk_filename,
                    "-C", "config.lmfb.40ch"
                ])

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

    np.save("mean.npy", mean)
    np.save("var.npy", var)


def preprocess(raw_dir, processed_dir):
    logger.info("Write outputs to file")
    outputs = {'train': [], 'test': []}
    mean = np.load("mean.npy")
    var = np.load("var.npy")
    vocab_word = load_tkn_to_idx(os.path.join(processed_dir, "vocab_words.txt"))
    vocab_char = load_tkn_to_idx(os.path.join(processed_dir, "vocab_chars.txt"))

    for mode in ["test", "train"]:
        os.makedirs(os.path.join(processed_dir, mode, "npy"), exist_ok=True)
        with open(os.path.join(raw_dir, mode, "prompts.txt"), encoding="utf-8") as f:
            lines = f.read().split("\n")
            for i, s in tqdm(list(enumerate(lines)), desc=mode):
                filename = s.split(' ')[0]
                if filename == "":
                    continue
                npy_filename = os.path.join(processed_dir, mode, "npy", filename + ".npy")

                if True:
                    # (rate, sig) = wav.read(wav_filename)
                    htk_filename = os.path.join(processed_dir, mode, "features", filename + ".htk")
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

                trans = s.lower().split(' ', 1)[1].replace(':', '')
                outputs[mode].append(dict(
                    filename=npy_filename,
                    target_word=' '.join([str(get_token_id(vocab_word, normalize_word(word)))
                                          for word in trans.split(' ')]),
                    target_char=' '.join([str(get_token_id(vocab_char, normalize_char(c).replace(' ', '_'))) for c in trans]),
                    trans_words=' '.join(s.lower().split(' ')[1:])
                ))

    for mode in ["test", "train"]:
        #outputs[mode].sort(key=lambda item: len(item['target_word']))
        for unit in ["word", "char"]:
            logger.info("Output to %s" % os.path.join(processed_dir, "%s_%s" % (unit, mode) + '.csv'))
            with open(os.path.join(processed_dir, "%s_%s" % (unit, mode) + '.csv'), 'w', encoding='utf-8') as f:
                f.write('\t'.join(['sound', 'target', 'trans']) + '\n')
                for o in outputs[mode]:
                    f.write('\t'.join([
                        o['filename'],
                        o['target_%s' % unit],
                        o['trans_words']
                    ]) + '\n')


class VIVOS(VoiceDataset):
    input_size = 120

    def __init__(self, mode, params):
        super().__init__(mode, params)
        cfg = params.dataset
        self._vocab = Vocab(os.path.join(self.get_processed_data_dir(), "vocab_%ss.txt" % cfg.unit))

        is_debug = mode == "debug"
        if mode == "debug":
            mode = "train"

        with open(
                os.path.join(self.get_processed_data_dir(), "%s_%s" % (cfg.unit, mode) + '.csv'),
                'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
            lines = [l.split('\t') for l in lines if l != ""]
            self.data = [{
                'X_path': l[0],
                'Y': [int(w) for w in l[1].split(' ')],
                'Y_len': len(l[1].split(' '))
            } for l in lines]

            if is_debug:
                self.data = self.data[:24]

        self._use_sos_eos = False
        self.insert_sos_eos_tokens()
        # self.insert_blank_token()

    def insert_sos_eos_tokens(self):
        self._vocab.add_token('<sos>')
        self._vocab.add_token('<eos>')
        self._use_sos_eos = True
        for item in self.data:
            item['Y'] = [self._vocab['<sos>']] + item['Y'] + [self._vocab['<eos>']]
            item['Y_len'] = len(item['Y'])

    def insert_blank_token(self):
        self._vocab.add_token('<blank>')

    @property
    def sos_token_id(self):
        return self._vocab['<sos>']

    @property
    def eos_token_id(self):
        return self._vocab['<eos>'] or self.pad_token_id

    @property
    def pad_token_id(self):
        return self._vocab['<pad>']

    @property
    def blank_token_id(self):
        return self._vocab['<blank>']

    @property
    def output_size(self):
        return len(self._vocab)

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        super().maybe_download_and_extract(force)
        if not os.path.exists(cls.get_raw_data_dir()):
            cls.download_and_extract(DOWNLOAD_URL, cls.get_raw_data_dir())

    @classmethod
    def maybe_preprocess(cls, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(cls.get_processed_data_dir()):
            return
        raw_dir = os.path.join(cls.get_raw_data_dir(), "vivos")
        os.makedirs(cls.get_processed_data_dir(), exist_ok=True)
        build_vocab(raw_dir, cls.get_processed_data_dir())
        calculate_mean_and_var(raw_dir, cls.get_processed_data_dir())
        preprocess(raw_dir, cls.get_processed_data_dir())

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
            pr = np.array(pr)
            gt = gt.cpu().detach().numpy()
            gt = gt[:gt_len] if not self._use_sos_eos else gt[1:gt_len - 1]
            dist += nltk.edit_distance(pr, gt)
            count += len(gt)
        return dist, count

    def format_output(self, y_pred, batch_input):
        pr = np.array(y_pred)
        gt = batch_input['Y'].cpu().detach().numpy()
        gt = gt[:batch_input['Y_len']] if not self._use_sos_eos else gt[1:batch_input['Y_len'] - 1]
        if self.params.dataset.output_format is None:
            return "", str(gt), str(pr)
        elif self.params.dataset.output_format == "text":
            delimiter = ' ' if self.params.dataset.unit == "word" else ''
            return \
                batch_input['X_path'], \
                delimiter.join(self._vocab.decode_idx_list(gt)), \
                delimiter.join(self._vocab.decode_idx_list(pr))
