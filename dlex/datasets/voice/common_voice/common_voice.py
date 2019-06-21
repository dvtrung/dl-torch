import os

import pandas

from dlex.datasets.voice.torch import PytorchVoiceDataset
from dlex.datasets.voice.builder import VoiceDatasetBuilder
from dlex.datasets.nlp.utils import write_vocab, normalize_string, char_tokenize, space_tokenize
from dlex.utils.logging import logger


def build_vocab(raw_dir, processed_dir):
    df = pandas.read_csv(os.path.join(raw_dir, "train.tsv"), sep='\t')
    write_vocab(processed_dir, df['sentence'], name="words", normalize_fn=normalize_string, tokenize_fn=space_tokenize)
    write_vocab(processed_dir, df['sentence'], name="chars", normalize_fn=normalize_string, tokenize_fn=char_tokenize)


class CommonVoiceBuilder(VoiceDatasetBuilder):
    def __init__(self, params):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        self.download_and_extract(
            "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-2/en.tar.gz",
            self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(self.get_processed_data_dir()):
            return
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)

        build_vocab(self.get_raw_data_dir(), self.get_processed_data_dir())

        # dfs = {mode: pandas.read_csv(os.path.join(self.get_raw_data_dir(), "%s.tsv" % mode), sep='\t') for mode in ['train', 'test']}

        dfs = pandas.read_csv(os.path.join(self.get_raw_data_dir(), "validated.tsv"), sep='\t')
        dfs = {'train': dfs[10000:], 'test': dfs[:10000]}
        file_paths = {mode: [
            os.path.join(self.get_raw_data_dir(), "clips", r['path'] + ".mp3") for _, r in dfs[mode].iterrows()
        ] for mode in ['train', 'test']}
        transcripts = {mode: [
            r['sentence'] for _, r in dfs[mode].iterrows() if r['sentence'] is not None and isinstance(r['sentence'], str)
        ] for mode in ['train', 'test']}
        self.extract_features(file_paths)
        self.regularize(file_paths)
        for token_type in ['word', 'char']:
            self.write_dataset(
                token_type,
                file_paths,
                transcripts,
                vocab_path=os.path.join(self.get_processed_data_dir(), "vocab", f"{token_type}s.txt"),
                normalize_fn=normalize_string,
                tokenize_fn=space_tokenize if token_type == 'word' else char_tokenize
            )

    def get_pytorch_wrapper(self, mode: str):
        return PytorchCommonVoice(self, mode, self._params)


class PytorchCommonVoice(PytorchVoiceDataset):
    input_size = 120

    def __init__(self, builder, mode, params):
        super().__init__(
            builder, mode, params,
            vocab_path=os.path.join(builder.get_processed_data_dir(), "vocab", "%ss.txt" % params.dataset.unit))
        cfg = params.dataset

        is_debug = mode == "debug"
        if mode == "debug":
            mode = "train"

        with open(
                os.path.join(builder.get_processed_data_dir(), "%s_%s" % (cfg.unit, mode) + '.csv'),
                'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
            lines = [l.split('\t') for l in lines if l != ""]
            self._data = [{
                'X_path': l[0],
                'Y': [int(w) for w in l[1].split(' ')],
            } for l in lines]

            if is_debug:
                self._data = self._data[:20]