import os

import pandas
from tqdm import tqdm

from dlex.datasets.nlp.utils import normalize_string, write_vocab, Vocab
from dlex.utils.logging import logger
from .nmt import NMTBaseDataset

DOWNLOAD_URLS = {
    ('fra', 'eng'): "https://www.manythings.org/anki/fra-eng.zip",
    ('spa', 'eng'): "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
}


def filter_pair(p, max_length=10):
    return len(p[0]) < max_length and \
           len(p[1]) < max_length


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


class Tatoeba(NMTBaseDataset):
    def __init__(self, mode, params):
        cfg = params.dataset
        self.dataset_name = (params.dataset.source if params.dataset.source != 'eng' else params.dataset.target) + '-eng'
        vocab_paths = {
            cfg.source: os.path.join(self.get_processed_data_dir(), self.dataset_name, "vocab", cfg.source + ".txt"),
            cfg.target: os.path.join(self.get_processed_data_dir(), self.dataset_name, "vocab", cfg.target + ".txt")}
        super().__init__(mode, params, vocab_paths=vocab_paths)

    def _load_data(self):
        """
        :rtype: list[dict[str, Any]]
        """
        # Load data
        if self.mode in ["test", "train"]:
            data = []
            df = pandas.read_csv(os.path.join(self.get_processed_data_dir(), self.dataset_name, self.mode + ".csv"), sep='\t')
            for _, row in tqdm(list(df.iterrows()), desc=self.mode):
                data.append(dict(
                    X=[self.vocab[self.lang_src].sos_token_idx] +
                      [int(i) for i in row[self.lang_src].split(' ')] +
                      [self.vocab[self.lang_src].eos_token_idx],
                    Y=[self.vocab[self.lang_tgt].sos_token_idx] +
                      [int(i) for i in row[self.lang_tgt].split(' ')] +
                      [self.vocab[self.lang_tgt].eos_token_idx]
                ))
            return data
        elif self.mode == "infer":
            return []

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        super().maybe_download_and_extract(force)
        if not os.path.exists(cls.get_raw_data_dir()):
            for lang_pairs in DOWNLOAD_URLS:
                try:
                    cls.download_and_extract(DOWNLOAD_URLS[lang_pairs], cls.get_raw_data_dir())
                except Exception as e:
                    logger.error("Failed to download %s" % '-'.join(lang_pairs))
                    logger.error(str(e))

    @classmethod
    def maybe_preprocess(cls, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(cls.get_processed_data_dir()):
            return

        for lang_pairs in DOWNLOAD_URLS:
            try:
                dataset_name = "-".join(lang_pairs)
                filepath = os.path.join(cls.get_working_dir(), "raw", dataset_name, "%s.txt" % lang_pairs[0])
                logger.info("Reading data from %s" % filepath)

                # Read the file and split into lines
                lines = open(filepath, encoding='utf-8'). \
                    read().strip().split('\n')

                # Split every line into pairs and normalize
                pairs = [[normalize_string(s).split(' ') for s in l.split('\t')] for l in lines]
                pairs = [list(reversed(p)) for p in pairs]
                
                logger.info("Read %s sentence pairs", len(pairs))
                pairs = filter_pairs(pairs)
                logger.info("Trimmed to %s sentence pairs", len(pairs))

                os.makedirs(cls.get_processed_data_dir(), exist_ok=True)
                default_words = ['<pad>', '<sos>', '<eos>', '<oov>']

                vocab = {}
                for i in [0, 1]:
                    write_vocab(
                        os.path.join(cls.get_processed_data_dir(), dataset_name),
                        [_p[i] for _p in pairs],
                        output_file_name=lang_pairs[i],
                        min_freq=0,
                        specials=default_words)
                    vocab[lang_pairs[i]] = Vocab(
                        os.path.join(cls.get_processed_data_dir(), dataset_name, "vocab", lang_pairs[i] + ".txt"))

                data = {
                    'train': pairs[10000:],
                    'test': pairs[:10000]
                }
                for mode in ['train', 'test']:
                    with open(os.path.join(cls.get_processed_data_dir(), dataset_name, "%s.csv" % mode), 'w') as fo:
                        fo.write('\t'.join(list(lang_pairs) + [l + '-original' for l in lang_pairs]) + '\n')
                        for item in data[mode]:
                            fo.write('\t'.join([
                                ' '.join([str(vocab[lang_pairs[0]][w]) for w in item[0]]),
                                ' '.join([str(vocab[lang_pairs[1]][w]) for w in item[1]]),
                                ' '.join([w for w in item[0]]),
                                ' '.join([w for w in item[1]])
                            ]) + "\n")
            except Exception as e:
                logger.error("Failed to process %s" % '-'.join(lang_pairs))
                logger.error(str(e))
