"""NLP Dataset"""

import os
from typing import List, Dict, Callable

import nltk
import numpy as np
from tqdm import tqdm

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.nlp.utils import Vocab
from dlex.utils.logging import logger, beautify
from .utils import read_htk, wav2htk, audio2wav


class VoiceDataset(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)
        self._mean = None
        self._variance = None

    def _get_wav_path(self, original_path, prefix=None):
        file_name = os.path.basename(original_path)
        file_name, file_ext = os.path.splitext(file_name)
        prefix = "" if prefix is None else prefix + "_"
        return os.path.join(self.get_processed_data_dir(), "wav", "%s%s.wav" % (prefix, file_name))

    def _get_htk_path(self, original_path, prefix=None):
        file_name = os.path.basename(original_path)
        file_name, file_ext = os.path.splitext(file_name)
        prefix = "" if prefix is None else prefix + "_"
        if file_ext == "htk":
            return original_path
        else:
            return os.path.join(self.get_processed_data_dir(), "htk", "%s%s.htk" % (prefix, file_name))

    def _get_npy_path(self, original_path, prefix=None):
        file_name = os.path.basename(original_path)
        file_name, file_ext = os.path.splitext(file_name)
        prefix = "" if prefix is None else prefix + "_"
        return os.path.join(self.get_processed_data_dir(), "npy", "%s%s.npy" % (prefix, file_name))

    def extract_features(
            self,
            file_paths: Dict[str, List[str]]):
        """
        Extract features and calculate mean and var
        :param file_paths: {'train': list, 'test': list} filenames
        """
        logger.info("Extracting features...")
        # get mean
        num_filters = self.params.dataset.feature.num_filters or 120
        mean_path = os.path.join(self.get_processed_data_dir(), "mean.npy")
        var_path = os.path.join(self.get_processed_data_dir(), "var.npy")
        if os.path.exists(mean_path) and os.path.exists(var_path):
            return
        mean = np.array([0] * num_filters)
        var = np.array([0] * num_filters)
        count = 0

        f_trash = open(os.devnull, "w")
        for mode in file_paths.keys():
            os.makedirs(os.path.join(self.get_processed_data_dir(), "wav"), exist_ok=True)
            os.makedirs(os.path.join(self.get_processed_data_dir(), "htk"), exist_ok=True)
            os.makedirs(os.path.join(self.get_processed_data_dir(), "npy"), exist_ok=True)

            for file_path in tqdm(file_paths[mode], desc=mode):
                _, file_ext = os.path.splitext(file_path)

                try:
                    # convert to wav
                    if file_ext == ".mp3":
                        wav_path = self._get_wav_path(file_path)
                        audio2wav(file_path, wav_path)
                    elif file_ext == ".wav":
                        wav_path = file_path
                    elif file_ext == ".htk":
                        pass
                    else:
                        raise Exception("Unsupported file type %s" % file_ext)

                    feat = None
                    if self.params.dataset.feature.tool == "htk":
                        # export feature
                        htk_path = self._get_htk_path(file_path)
                        wav2htk(wav_path, htk_path)
                        feat = read_htk(htk_path)
                    elif self.params.dataset.feature.tool == "librosa":
                        raise Exception("Not implemented.")
                    elif self.params.dataset.feature.tool == "python_speech_features":
                        npy_path = self._get_npy_path(file_path)
                        if not os.path.exists(npy_path):
                            from python_speech_features import logfbank, delta
                            import scipy.io.wavfile as wav
                            try:
                                (rate, sig) = wav.read(wav_path)
                                feat = logfbank(sig, rate, winlen=0.025, winstep=0.01, nfilt=40)
                                d1 = delta(feat, 2)
                                d2 = delta(d1, 2)
                                feat = np.concatenate([feat, d1, d2], axis=-1)
                                np.save(npy_path, feat)
                            except Exception as e:
                                logger.error(str(e))
                        else:
                            feat = np.load(npy_path)

                    # update mean and var
                    if mode == "train":
                        if feat is not None:
                            for k in range(len(feat)):
                                updated_mean = (mean * count + feat[k]) / (count + 1)
                                var = (count * var + (feat[k] - mean) * (feat[k] - updated_mean)) / (count + 1)
                                mean = updated_mean
                                count += 1
                except FileExistsError as e:
                    logger.error("Error processing %s (%s)", file_path, str(e))
        f_trash.close()
        logger.debug("mean: %s", beautify(mean))
        logger.debug("var: %s", beautify(var))
        np.save(mean_path, mean)
        np.save(var_path, var)

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.load(os.path.join(self.get_processed_data_dir(), "mean.npy"))
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            self._variance = np.load(os.path.join(self.get_processed_data_dir(), "var.npy"))
        return self._variance

    def load_feature(self, path: str, regularize=True):
        if self.params.dataset.feature.file_type == "npy":
            dat = np.load(path)
        elif self.params.dataset.feature.file_type == "htk":
            dat = read_htk(path)

        if dat is None:
            return None
        else:
            return dat if not regularize else (dat - self.mean) / np.sqrt(self.variance)

    def write_dataset(
            self,
            output_prefix: str,
            file_paths: Dict[str, List[str]],
            transcripts: Dict[str, List[str]],
            vocab_path: str,
            normalize_fn: Callable[[str], str],
            tokenize_fn: Callable[[str], List[str]]):
        processed_dir = self.get_processed_data_dir()
        vocab = Vocab(vocab_path)
        for mode in file_paths.keys():
            outputs = []
            output_fn = os.path.join(processed_dir, "%s_%s" % (output_prefix, mode) + '.csv')
            for file_path, transcript in tqdm(list(zip(file_paths[mode], transcripts[mode])), desc=mode):
                if file_path == "":
                    continue
                if self.params.dataset.feature.file_type == "npy":
                    feature_path = self._get_npy_path(file_path)
                elif self.params.dataset.feature.file_type == "htk":
                    feature_path = self._get_htk_path(file_path)
                else:
                    raise Exception("Feature file type not supported: %s" % self.params.dataset.feature.file_type)

                try:
                    tokens = [str(vocab[tkn]) for tkn in tokenize_fn(normalize_fn(transcript))]
                    if self.params.dataset.max_target_length is not None and \
                            len(tokens) > self.params.dataset.max_target_length:
                        continue
                    #if self.params.dataset.max_source_length is not None:
                    #    feat = self.load_feature(feature_path)
                    #    if len(feat) > self.params.dataset.max_source_length:
                    #        continue
                except FileNotFoundError:
                    logger.error("File '%s' does not exist." % feature_path)
                except Exception:
                    logger.error("Error reading '%s'." % feature_path)
                else:
                    outputs.append(dict(
                        filename=feature_path,
                        target=' '.join(
                            [str(vocab[tkn]) for tkn in tokenize_fn(normalize_fn(transcript))]),
                        original=transcript,
                        tokenized=' '.join(tokenize_fn(normalize_fn(transcript)))
                    ))

            # outputs[mode].sort(key=lambda item: len(item['target_word']))
            logger.info("Output to %s" % output_fn)
            with open(output_fn, 'w', encoding='utf-8') as f:
                f.write('\t'.join(['sound', 'target', 'original', 'tokenized']) + '\n')
                for o in outputs:
                    f.write('\t'.join([
                        o['filename'],
                        o['target'],
                        o['original'],
                        o['tokenized']
                    ]) + '\n')

    def evaluate(self, pred, ref, metric: str):
        if metric == "wer":
            return nltk.edit_distance(pred, ref), len(ref)