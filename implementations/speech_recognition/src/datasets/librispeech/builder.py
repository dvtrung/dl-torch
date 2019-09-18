import os
from pathlib import Path

from dlex.datasets.nlp.utils import char_tokenize, write_vocab, spacy_tokenize, \
    normalize_lower
from dlex.datasets.voice.builder import VoiceDataset
from dlex.utils import run_script
from dlex.utils.logging import logger

FILES = [
    # "raw-metadata.tar.gz",
    # "train-clean-100.tar.gz",
    "dev-clean.tar.gz",
    "test-clean.tar.gz",
    "train-other-500.tar.gz",
    "dev-other.tar.gz",
    "test-other.tar.gz"
]
BASE_URL = "http://www.openslr.org/resources/12/"


class LibriSpeech(VoiceDataset):
    input_size = 40

    def __init__(self, params):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        if not os.path.exists(self.get_raw_data_dir()):
            for f in FILES:
                self.download_and_extract(BASE_URL + f, os.path.join(self.get_raw_data_dir()))

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        #if os.path.exists(self.get_processed_data_dir()):
        #    return
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)

        working_dir = os.path.join(self.get_raw_data_dir(), "LibriSpeech")
        if not Path(self.get_processed_data_dir(), "wav").exists():
            logger.info("Converting mp3 files to wav...")
            run_script('convert-to-wav.py', [
                '-i', os.path.join(self.get_raw_data_dir(), "LibriSpeech"),
                '-o', os.path.join(self.get_processed_data_dir(), "wav"),
                '--num_workers', '4'])

        #if not Path(self.get_processed_data_dir(), "htk").exists():
        #    logger.info("Extracting features...")
        #    run_script('extract-htk-features.py', [
        #        '-i', os.path.join(self.get_processed_data_dir(), "wav"),
        #        '-o', os.path.join(self.get_processed_data_dir(), "htk"),
        #        '--num_workers', '4'])

        file_paths = {'train': [], 'valid': [], 'test': []}
        transcripts = {'train': [], 'valid': [], 'test': []}
        folders = {'train': 'train-other-500', 'valid': 'dev-other', 'test': 'test-other'}
        for mode in file_paths.keys():
            folder = folders[mode]
            for path in list(Path(working_dir, folder).glob("**/*.txt")):
                for line in path.open():
                    idx, transcript = line.split(' ', 1)
                    transcripts[mode].append(transcript.strip())
                    file_paths[mode].append(os.path.join(
                        self.get_processed_data_dir(),
                        "wav",
                        # {'train': 'train-clean-100', 'valid': 'dev-clean', 'test': 'test-clean'}[mode],
                        # *idx.split('-')[:-1],
                        idx + ".wav"
                    ))

        self.extract_features(file_paths)
        for token_type in ['word', 'char']:
            write_vocab(
                self.get_processed_data_dir(),
                transcripts['train'],
                output_file_name="%s.txt" % token_type,
                normalize_fn=normalize_lower,
                tokenize_fn=spacy_tokenize if token_type == "word" else char_tokenize)

            self.write_dataset(
                token_type,
                file_paths,
                transcripts,
                vocab_path=os.path.join(self.get_processed_data_dir(), "vocab", f"{token_type}.txt"),
                normalize_fn=normalize_lower,
                tokenize_fn=spacy_tokenize if token_type == 'word' else char_tokenize
            )

    def get_pytorch_wrapper(self, mode: str):
        from .torch import PytorchLibriSpeech
        return PytorchLibriSpeech(self, mode)
