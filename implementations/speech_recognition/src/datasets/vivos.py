import os

import numpy as np

from dlex.datasets.nlp.utils import write_vocab, char_tokenize, space_tokenize, normalize_none, Vocab
from dlex.datasets.voice.torch import PytorchSeq2SeqDataset
from dlex.datasets.voice.builder import VoiceDatasetBuilder
from dlex.torch import BatchItem

DOWNLOAD_URL = "https://ailab.hcmus.edu.vn/assets/vivos.tar.gz"


class VIVOS(VoiceDatasetBuilder):
    def __init__(self, params):
        super().__init__(params)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchVIVOS(self, mode, self._params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        if not os.path.exists(self.get_raw_data_dir()):
            self.download_and_extract(DOWNLOAD_URL, self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(self.get_processed_data_dir()):
            return
        raw_dir = os.path.join(self.get_raw_data_dir(), "vivos")
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)

        file_paths = {'train': [], 'test': []}
        transcripts = {'train': [], 'test': []}
        for mode in ['train', 'test']:
            with open(os.path.join(raw_dir, "train", "prompts.txt"), encoding="utf-8") as f:
                for s in f.read().split('\n'):
                    s = s.replace(':', '')
                    filename, sent = s.split(' ', 1)
                    file_path = os.path.join(raw_dir, mode, "waves", filename.split('_')[0], filename + ".wav")
                    file_paths[mode].append(file_path)
                    transcripts[mode].append(sent)

        write_vocab(self.get_processed_data_dir(), transcripts, name="words", normalize_fn=normalize_none, tokenize_fn=space_tokenize)
        write_vocab(self.get_processed_data_dir(), transcripts, name="chars", normalize_fn=normalize_none, tokenize_fn=char_tokenize)
        self.extract_features(file_paths)
        self.regularize(file_paths)
        for token_type in ['word', 'char']:
            self.write_dataset(
                token_type,
                file_paths,
                transcripts,
                vocab_path=os.path.join(self.get_processed_data_dir(), "vocab", f"{token_type}s.txt"),
                normalize_fn=normalize_none,
                tokenize_fn=space_tokenize if token_type == 'word' else char_tokenize
            )


class PytorchVIVOS(PytorchSeq2SeqDataset):
    input_size = 120

    def __init__(self, builder, mode, params):
        super().__init__(builder, mode, params)
        cfg = params.dataset
        self._vocab = Vocab(os.path.join(builder.get_processed_data_dir(), "vocab_%ss.txt" % cfg.unit))

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
            if cfg.sort:
                self._data.sort(key=lambda it: it['Y_len'])

            if is_debug:
                self._data = self._data[:cfg.debug_size]

    def __getitem__(self, i: int):
        item = self._data[i]
        X = np.load(item['X_path'])
        return BatchItem(X=X, Y=item['Y'])