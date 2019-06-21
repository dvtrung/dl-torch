"""Datasets for neural machine translation"""

import os
import shutil
from pathlib import Path

import nltk
import numpy as np
import torch
from tqdm import tqdm

from dlex.configs import ModuleConfigs
from torch.datasets import BaseDataset
from dlex.utils.ops_utils import LongTensor
from dlex.utils.utils import maybe_download, maybe_unzip

DOWNLOAD_URL = "http://datashare.is.ed.ac.uk/download/DS_10283_2791.zip"


class NSDTSEA(BaseDataset):
    working_dir = os.path.join(ModuleConfigs.DATA_TMP_PATH, "nsdtsea")
    raw_data_dir = os.path.join(working_dir, "raw")
    processed_data_dir = os.path.join(working_dir, "data")

    def __init__(self, mode, params):
        super().__init__(mode, params)

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        super().maybe_download_and_extract(force)
        if os.path.exists(cls.raw_data_dir):
            return
        maybe_download(
            "data.zip",
            cls.working_dir,
            DOWNLOAD_URL)
        maybe_unzip("data.zip", cls.working_dir, "raw")
        for zip_name in [
                "clean_testset_wav",
                "clean_trainset_28spk_wav",
                "clean_trainset_56spk_wav",
                "noisy_testset_wav",
                "noisy_trainset_28spk_wav",
                "noisy_trainset_56spk_wav"]:
            maybe_unzip(zip_name + ".zip",
                        cls.raw_data_dir,
                        zip_name)
        shutil.rmtree(os.path.join(cls.working_dir, "data.zip"))

    @classmethod
    def maybe_preprocess(cls, force=False):
        super().maybe_preprocess(force)
        file_names = Path(cls.raw_data_dir, "rnnoise_contributions").glob("*.raw")
        for fn in tqdm(list(file_names)):
            data = np.memmap(fn)
            import wave
            with wave.open(str(fn)[:-4] + '.wav', 'wb') as wavfile:
                wavfile.setparams((2, 2, 44100, 0, 'NONE', 'NONE'))
                wavfile.writeframes(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        batch.sort(key=lambda item: len(item['X']), reverse=True)
        inp = [LongTensor(item['X']).view(-1) for item in batch]
        tgt = [LongTensor(item['Y']).view(-1) for item in batch]
        inp = torch.nn.utils.rnn.pad_sequence(
            inp, batch_first=True,
            padding_value=self.word_to_idx[self.lang[0]]["<eos>"])
        tgt = torch.nn.utils.rnn.pad_sequence(
            tgt, batch_first=True,
            padding_value=self.word_to_idx[self.lang[1]]["<eos>"])

        return dict(
            X=inp, X_len=LongTensor([len(item['X']) for item in batch]),
            Y=tgt, Y_len=LongTensor([len(item['Y']) for item in batch]))

    def evaluate(self, y_pred, batch, metric):
        if metric == "bleu":
            target_variables = batch.Y
            score, total = 0, 0
            for k, _y_pred in enumerate(y_pred):
                target = self._trim_result(target_variables[k].cpu().detach().numpy().tolist())
                predicted = self._trim_result(_y_pred)
                score += nltk.translate.bleu_score.sentence_bleu([target], predicted)
                total += 1
            return score, total

    def format_output(self, y_pred, batch_item):
        src = self._trim_result(batch_item['X'].cpu().numpy())
        tgt = self._trim_result(batch_item['Y'].cpu().numpy())
        y_pred = self._trim_result(y_pred)
        if self.cfg.output_format == "text":
            return ' '.join([self.idx_to_word[self.lang_src][word_id] for word_id in src]), \
                   ' '.join([self.idx_to_word[self.lang_tgt][word_id] for word_id in tgt]), \
                   ' '.join([self.idx_to_word[self.lang_tgt][word_id] for word_id in y_pred])
        else:
            return super().format_output(y_pred, batch_item)