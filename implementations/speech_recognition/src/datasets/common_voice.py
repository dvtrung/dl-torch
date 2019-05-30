import os

import pandas

from dlex.datasets.base.nlp import write_vocab
from dlex.datasets.base.voice import extract_features, regularize, write_dataset, SpeechRecognitionDataset
from dlex.utils.logging import logger


def build_vocab(raw_dir, processed_dir):
    df = pandas.read_csv(os.path.join(raw_dir, "train.tsv"), sep='\t')
    write_vocab(processed_dir, df['sentence'], name="words", token='word')
    write_vocab(processed_dir, df['sentence'], name="chars", token='char')
    logger.info("Vocab written to %s", processed_dir)


def preprocess(raw_dir, processed_dir):
    dfs = {mode: pandas.read_csv(os.path.join(raw_dir, "%s.tsv" % mode), sep='\t') for mode in ['train', 'test']}
    file_paths = {mode: [
        os.path.join(raw_dir, "clips", r['path'] + ".mp3") for _, r in dfs[mode].iterrows()
    ] for mode in ['train', 'test']}
    transcripts = {mode: [
        r['sentence'] for _, r in dfs[mode].iterrows()
    ] for mode in ['train', 'test']}
    extract_features(
        processed_dir,
        file_paths,
    )
    regularize(
        processed_dir,
        file_paths,
        transcripts,
    )
    write_dataset(
        processed_dir,
        file_paths,
        transcripts,
        vocab_word_path=os.path.join(processed_dir, "vocab", "words.txt"),
        vocab_char_path=os.path.join(processed_dir, "vocab", "chars.txt"),
    )


class CommonVoice(SpeechRecognitionDataset):
    input_size = 120

    def __init__(self, mode, params):
        super().__init__(
            mode, params,
            vocab_path=os.path.join(self.get_processed_data_dir(), "vocab", "%ss.txt" % params.dataset.unit))
        cfg = params.dataset

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
                'Y': [self.sos_token_id] + [int(w) for w in l[1].split(' ')] + [self.eos_token_id],
                'Y_len': len(l[1].split(' ')) + 2
            } for l in lines]

            if is_debug:
                self.data = self.data[:20]

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        super().maybe_download_and_extract(force)

    @classmethod
    def maybe_preprocess(cls, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(cls.get_processed_data_dir()):
            return
        os.makedirs(cls.get_processed_data_dir(), exist_ok=True)

        build_vocab(cls.get_raw_data_dir(), cls.get_processed_data_dir())
        preprocess(cls.get_raw_data_dir(), cls.get_processed_data_dir())