import os

import pandas

from dlex.datasets.voice.torch import PytorchVoiceDataset
from dlex.datasets.voice.builder import VoiceDataset
from dlex.datasets.nlp.utils import write_vocab, normalize_string, char_tokenize, space_tokenize
from dlex.utils.logging import logger
from dlex.utils.utils import run_script


class CommonVoice(VoiceDataset):
    def __init__(self, params):
        super().__init__(params)
        self._file_paths = None
        self._transcripts = None

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        self.download_and_extract(
            "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-2/en.tar.gz",
            self.get_raw_data_dir())

    def _get_raw_data(self):
        """Loading raw data may take a long time and is only loaded when necessary."""
        if self._file_paths is not None:
            return self._file_paths, self._transcripts
        # Load raw data
        logger.info("Loading dataset...")
        dfs = pandas.read_csv(os.path.join(self.get_raw_data_dir(), "validated.tsv"), sep='\t')
        dfs = {'train': dfs[10000:], 'test': dfs[:10000]}

        htk_root = os.path.join(self.get_processed_data_dir(), "htk")
        file_paths = {mode: [
            os.path.join(htk_root, r['path'] + ".htk") for _, r in dfs[mode].iterrows()
            if r['sentence'] is not None and isinstance(r['sentence'], str)
        ] for mode in ['train', 'test']}
        transcripts = {mode: [
            r['sentence'] for _, r in dfs[mode].iterrows() if
            r['sentence'] is not None and isinstance(r['sentence'], str)
        ] for mode in ['train', 'test']}
        self._file_paths = file_paths
        self._transcripts = transcripts
        return file_paths, transcripts

    @property
    def output_prefix(self):
        cfg = self.params.dataset
        return "%s_max_length_%d_%d_vocab_size_%d" % (
            cfg.unit,
            cfg.max_source_length or 0,
            cfg.max_target_length or 0,
            cfg.vocab_size or 0)

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        # if os.path.exists(self.get_processed_data_dir()):
        #     return
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)

        for token_type in ['char', 'word']:
            vocab_path = os.path.join(self.get_processed_data_dir(), "vocab", f"{token_type}.txt")
            if os.path.exists(vocab_path):
                continue
            file_paths, transcripts = self._get_raw_data()
            write_vocab(
                self.get_processed_data_dir(),
                transcripts['train'],
                output_file_name="%s.txt" % token_type,
                normalize_fn=normalize_string,
                tokenize_fn=space_tokenize if token_type == "word" else char_tokenize)

        htk_root = os.path.join(self.get_processed_data_dir(), "htk")
        wav_root = os.path.join(self.get_processed_data_dir(), "wav")

        if not (os.path.exists(wav_root) and len(os.listdir(wav_root)) > 0):
            logger.info("Converting mp3 files to wav...")
            run_script('convert-to-wav.py', [
                '-i', os.path.join(self.get_raw_data_dir(), "clips"),
                '-o', os.path.join(self.get_processed_data_dir(), "wav"),
                '--num_workers', 4])

        if not (os.path.exists(htk_root) and len(os.listdir(htk_root)) > 0):
            logger.info("Extracting features...")
            run_script('extract-htk-features.py', [
                '-i', os.path.join(self.get_processed_data_dir(), "wav"),
                '-o', os.path.join(self.get_processed_data_dir(), "htk"),
                '--num_workers', 4])

            file_paths, transcripts = self._get_raw_data()
            self.extract_features(file_paths)

        for token_type in ['word', 'char']:
            output_prefix = self.output_prefix
            logger.info("Output prefix: %s", output_prefix)
            if not os.path.exists(os.path.join(self.get_processed_data_dir(), f"{output_prefix}_train.csv")) \
                    or not os.path.exists(os.path.join(self.get_processed_data_dir(), f"{output_prefix}_test.csv")):
                file_paths, transcripts = self._get_raw_data()
                self.write_dataset(
                    output_prefix,
                    file_paths,
                    transcripts,
                    vocab_path=os.path.join(self.get_processed_data_dir(), "vocab", f"{token_type}.txt"),
                    normalize_fn=normalize_string,
                    tokenize_fn=space_tokenize if token_type == 'word' else char_tokenize,
                )

    def get_pytorch_wrapper(self, mode: str):
        return PytorchCommonVoice(self, mode, self.params)


class PytorchCommonVoice(PytorchVoiceDataset):
    input_size = 120

    def __init__(self, builder, mode, params):
        super().__init__(
            builder, mode, params,
            vocab_path=os.path.join(builder.get_processed_data_dir(), "vocab", "%s.txt" % params.dataset.unit))

        is_debug = mode == "debug"
        if mode == "debug":
            mode = "train"

        self._data = self.load_data(os.path.join(
            builder.get_processed_data_dir(),
            "%s_%s" % (builder.output_prefix, mode) + '.csv'))

        if is_debug:
            self._data = self._data[:20]