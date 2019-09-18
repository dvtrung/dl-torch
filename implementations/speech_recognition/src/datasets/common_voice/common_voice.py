import os
from subprocess import call

import pandas

from dlex.datasets.nlp.utils import write_vocab, normalize_lower, char_tokenize, spacy_tokenize, \
    normalize_lower_alphanumeric
from dlex.datasets.voice.builder import VoiceDataset
from dlex.datasets.voice.torch import PytorchVoiceDataset
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

        wav_root = os.path.join(self.get_processed_data_dir(), "wav")
        file_paths = {mode: [
            os.path.join(wav_root, r['path'] + ".wav") for _, r in dfs[mode].iterrows()
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
        s = "%s_vocab_size_%d" % (
            cfg.unit,
            cfg.vocab_size or 0)
        if cfg.alphanumeric:
            s += "_alphanumeric"
        return s

    @property
    def vocab_path(self):
        return os.path.join(self.get_processed_data_dir(), "vocab", f"{self.params.dataset.unit}.txt")

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        # if os.path.exists(self.get_processed_data_dir()):
        #   return
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)

        if not os.path.exists(self.vocab_path):
            file_paths, transcripts = self._get_raw_data()
            write_vocab(
                self.get_processed_data_dir(),
                transcripts['train'],
                output_file_name="%s.txt" % self.params.dataset.unit,
                normalize_fn=normalize_lower,
                tokenize_fn=spacy_tokenize if self.params.dataset.unit == "word" else char_tokenize,
                min_freq=5 if self.params.dataset.unit == "char" else 0)

        npy_root = os.path.join(self.get_processed_data_dir(), "npy")
        wav_root = os.path.join(self.get_processed_data_dir(), "wav")

        if not (os.path.exists(wav_root) and len(os.listdir(wav_root)) > 0):
            logger.info("Converting mp3 files to wav...")
            run_script('convert-to-wav.py', [
                '-i', os.path.join(self.get_raw_data_dir(), "clips"),
                '-o', os.path.join(self.get_processed_data_dir(), "wav"),
                '--num_workers', "4"])

        if not (os.path.exists(npy_root) and len(os.listdir(npy_root)) > 0):
            logger.info("Extracting features...")
            #    run_script('extract-htk-features.py', [
            #        '-i', os.path.join(self.get_raw_data_dir(), "clips"),
            #        '-o', os.path.join(self.get_processed_data_dir(), "htk"),
            #        '--num_workers', "4"])

            file_paths, transcripts = self._get_raw_data()
            self.extract_features(file_paths)

        output_prefix = self.output_prefix
        logger.info("Output prefix: %s", output_prefix)
        if self.params.dataset.alphanumeric:
            normalize_fn = normalize_lower_alphanumeric
        else:
            normalize_fn = normalize_lower
        if not os.path.exists(os.path.join(self.get_processed_data_dir(), f"{output_prefix}_train.csv")) \
                or not os.path.exists(os.path.join(self.get_processed_data_dir(), f"{output_prefix}_test.csv")):
            file_paths, transcripts = self._get_raw_data()
            self.write_dataset(
                output_prefix,
                file_paths,
                transcripts,
                vocab_path=self.vocab_path,
                normalize_fn=normalize_fn,
                tokenize_fn=spacy_tokenize if self.params.dataset.unit == 'word' else char_tokenize,
            )

    def get_pytorch_wrapper(self, mode: str):
        return PytorchCommonVoice(self, mode)


class PytorchCommonVoice(PytorchVoiceDataset):
    input_size = 120

    def __init__(self, builder, mode):
        super().__init__(
            builder, mode,
            vocab_path=os.path.join(builder.get_processed_data_dir(), "vocab", "%s.txt" % builder.params.dataset.unit))
        if mode == "infer":
            self._data = []
        else:
            self._data = self.load_data(self.csv_path)

    @property
    def csv_path(self):
        return os.path.join(
            self.builder.get_processed_data_dir(),
            "%s_%s" % (self.builder.output_prefix, self.mode) + '.csv')

    def load_from_input(self, s):
        output_path = os.path.join("tmp", os.path.basename(s))
        f_trash = open(os.devnull, "w")
        call(
            ["ffmpeg", "-n", "-i", s, "-ar", "16000", "-ac", "1", output_path],
            stdout=f_trash, stderr=f_trash)
        self._data = [{
            'X': self.builder.regularize(self.builder.get_features_from_audio(output_path)),
            'Y': [0]
        }]