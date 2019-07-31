import os

from dlex.datasets.nlp.utils import write_vocab, char_tokenize, space_tokenize, normalize_char, \
    normalize_string
from dlex.datasets.voice.builder import VoiceDataset

DOWNLOAD_URL = "https://ailab.hcmus.edu.vn/assets/vivos.tar.gz"


class VIVOS(VoiceDataset):
    def __init__(self, params):
        super().__init__(params)

    def get_pytorch_wrapper(self, mode: str):
        from .torch import PytorchVIVOS
        return PytorchVIVOS(self, mode)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        self.download_and_extract(DOWNLOAD_URL, self.get_raw_data_dir())

    def get_vocab_path(self, unit):
        return os.path.join(self.get_processed_data_dir(), "vocab", "%ss.txt" % unit)

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        #if os.path.exists(self.get_processed_data_dir()):
        #    return
        raw_dir = os.path.join(self.get_raw_data_dir(), "vivos")
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)

        file_paths = {'train': [], 'test': []}
        transcripts = {'train': [], 'test': []}
        for mode in ['train', 'test']:
            with open(os.path.join(raw_dir, mode, "prompts.txt"), encoding="utf-8") as f:
                for s in f.read().split('\n'):
                    if s.strip() == "":
                        continue
                    s = s.replace(':', '')
                    filename, sent = s.split(' ', 1)
                    file_path = os.path.join(raw_dir, mode, "waves", filename.split('_')[0], filename + ".wav")
                    file_paths[mode].append(file_path)
                    transcripts[mode].append(sent)

        write_vocab(
            self.get_processed_data_dir(), transcripts['train'],
            output_file_name="words.txt",
            normalize_fn=normalize_string,
            tokenize_fn=space_tokenize)
        write_vocab(
            self.get_processed_data_dir(), transcripts['train'],
            output_file_name="chars.txt",
            normalize_fn=normalize_char,
            tokenize_fn=char_tokenize)

        self.extract_features(file_paths)
        for token_type in ['word', 'char']:
            self.write_dataset(
                token_type,
                file_paths,
                transcripts,
                vocab_path=os.path.join(self.get_processed_data_dir(), "vocab", f"{token_type}s.txt"),
                normalize_fn=normalize_string if token_type == 'word' else normalize_char,
                tokenize_fn=space_tokenize if token_type == 'word' else char_tokenize
            )