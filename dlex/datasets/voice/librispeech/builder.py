import os
import glob

import pandas

from dlex.datasets.voice.builder import VoiceDatasetBuilder
from dlex.datasets.nlp.utils import normalize_string, char_tokenize, space_tokenize
from dlex.utils.logging import logger
from utils import run_script

FILES = ["raw-metadata.tar.gz", "train-clean-100.tar.gz", "dev-clean.tar.gz"]
BASE_URL = "http://www.openslr.org/resources/12/"


class LibriSpeech(VoiceDatasetBuilder):
    input_size = 120

    def __init__(self, params):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        if not os.path.exists(self.get_raw_data_dir()):
            for f in FILES:
                self.download_and_extract(BASE_URL + f, os.path.join(self.get_raw_data_dir()))

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(self.get_processed_data_dir()):
            return
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)

        working_dir = os.path.join(self.get_raw_data_dir(), "LibriSpeech")
        for trans_path in glob.glob()

        logger.info("Converting mp3 files to wav...")
        run_script('convert-to-wav.py', [
            '-i', os.path.join(self.get_raw_data_dir(), "clips"),
            '-o', os.path.join(self.get_processed_data_dir(), "wav"),
            '--num_workers', 4])
        logger.info("Extracting features...")
        run_script('extract-htk-features.py', [
            '-i', os.path.join(self.get_processed_data_dir(), "wav"),
            '-o', os.path.join(self.get_processed_data_dir(), "htk"),
            '--num_workers', 4])

        dfs = pandas.read_csv(os.path.join(self.get_raw_data_dir(), "validated.tsv"), sep='\t')
        dfs = {'train': dfs[10000:], 'test': dfs[:10000]}
        file_paths = {mode: [
            os.path.join(self.get_processed_data_dir(), "htk", r['path'] + ".htk") for _, r in dfs[mode].iterrows()
        ] for mode in ['train', 'test']}
        transcripts = {mode: [
            r['sentence'] for _, r in dfs[mode].iterrows() if
            r['sentence'] is not None and isinstance(r['sentence'], str)
        ] for mode in ['train', 'test']}
        self.extract_features(file_paths)
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
        from .torch import PytorchLibriSpeech
        return PytorchLibriSpeech(self, mode, self.params)
