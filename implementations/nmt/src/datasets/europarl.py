import os

from tqdm import tqdm

from dlex.datasets.base.nlp import NLPDataset, load_tkn_to_idx, load_idx_to_tkn, normalize_string
from dlex.utils.logging import logger

DOWNLOAD_URLS = {
    ('cs', 'en'): 'https://www.statmt.org/europarl/v7/cs-en.tgz'
}


def read_lang(filepath):
    logger.info("Reading data from %s" % filepath)
    # Read the file and split into lines
    lines = open(filepath, encoding='utf-8'). \
        read().strip().split('\n')
    return lines


def filter_pair(p, max_length=10):
    return len(p[0]) < max_length and \
           len(p[1]) < max_length


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


class Europarl(NLPDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)
        cfg = params.dataset

        self.lang_src = cfg.source
        self.lang_tgt = cfg.target
        self.lang = (self.lang_src, self.lang_tgt)
        self.dataset_name = (self.lang_src if self.lang_src != 'eng' else self.lang_tgt) + '-eng'

        # Load vocab
        self.word2index = {}
        self.index2word = {}
        for lang in self.lang:
            self.word2index[lang] = load_tkn_to_idx(
                os.path.join(self.get_processed_data_dir(), self.dataset_name, "vocab", lang + ".txt"))
            self.index2word[lang] = load_idx_to_tkn(
                os.path.join(self.get_processed_data_dir(), self.dataset_name, "vocab", lang + ".txt"))
            logger.info("%s vocab size: %d", lang, len(self.word2index[lang]))

        self.input_size = len(self.word2index[self.lang_src])
        self.output_size = len(self.word2index[self.lang_tgt])

        # Load data
        if self.mode in ["test", "train"]:
            data = []
            fo = open(os.path.join(self.get_processed_data_dir(), self.dataset_name, self.mode + ".csv"), "r", encoding='utf-8')
            lang_pairs = fo.readline().strip().split('\t')[:2]
            for line in tqdm(fo):
                pairs = line.strip().split('\t')[:2]
                pairs = {lang: pairs[i] for i, lang in enumerate(lang_pairs)}
                data.append(dict(
                    X=[self.word2index[self.lang_src]['<sos>']] + [int(i) for i in pairs[self.lang_src].split(' ')] + [self.word2index[self.lang_src]['<eos>']],
                    Y=[self.word2index[self.lang_tgt]['<sos>']] + [int(i) for i in pairs[self.lang_tgt].split(' ')] + [self.word2index[self.lang_tgt]['<eos>']]
                ))
            fo.close()
            self.data = data
        elif self.mode == "infer":
            self.data = []

    @property
    def sos_id(self):
        return self.word2index[self.lang_src]['<sos>']

    @property
    def eos_id(self):
        return self.word2index[self.lang_src]['<eos>']

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        super().maybe_download_and_extract(force)
        if not os.path.exists(cls.get_raw_data_dir()):
            for lang_pairs in DOWNLOAD_URLS:
                try:
                    cls.download_and_extract(
                        DOWNLOAD_URLS[lang_pairs],
                        os.path.join(cls.get_raw_data_dir(), '-'.join(lang_pairs)))
                except:
                    logger.error("Failed to download %s" % '-'.join(lang_pairs))

    @classmethod
    def maybe_preprocess(cls, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(cls.get_processed_data_dir()):
            return

        for lang_pairs in DOWNLOAD_URLS:
            try:
                dataset_name = "-".join(lang_pairs)
                pairs = zip(
                    read_lang(os.path.join(cls.get_working_dir(), dataset_name, "europarl-v7.%s.%s" % (dataset_name, lang_pairs[0]))),
                    read_lang(os.path.join(cls.get_raw_data_dir(), dataset_name, "europarl-v7.%s.%s" % (dataset_name, lang_pairs[1]))))
                logger.info("Read %s sentence pairs", len(pairs))
                pairs = filter_pairs(pairs)
                logger.info("Trimmed to %s sentence pairs", len(pairs))

                os.makedirs(cls.get_processed_data_dir(), exist_ok=True)
                default_words = ['<pad>', '<sos>', '<eos>', '<oov>']

                word_token_to_idx = {}
                for i in [0, 1]:
                    prepare_vocab_words(
                        os.path.join(cls.get_processed_data_dir(), dataset_name),
                        [_p[i] for _p in pairs],
                        lang_pairs[i], 0, default_words)
                    word_token_to_idx[lang_pairs[i]] = load_tkn_to_idx(
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
                                ' '.join([str(get_token_id(word_token_to_idx[lang_pairs[0]], w)) for w in item[0]]),
                                ' '.join([str(get_token_id(word_token_to_idx[lang_pairs[1]], w)) for w in item[1]]),
                                ' '.join([w for w in item[0]]),
                                ' '.join([w for w in item[1]])
                            ]) + "\n")
            except:
                logger.error("Failed to process %s" % '-'.join(lang_pairs))

    def collate_fn(self, batch):
        batch.sort(key=lambda item: len(item['X']), reverse=True)
        inp = [LongTensor(item['X']).view(-1) for item in batch]
        tgt = [LongTensor(item['Y']).view(-1) for item in batch]
        inp = torch.nn.utils.rnn.pad_sequence(
            inp, batch_first=True,
            padding_value=self.word2index[self.lang[0]]["<eos>"])
        tgt = torch.nn.utils.rnn.pad_sequence(
            tgt, batch_first=True,
            padding_value=self.word2index[self.lang[1]]["<eos>"])

        return dict(
            X=inp, X_len=LongTensor([len(item['X']) for item in batch]),
            Y=tgt, Y_len=LongTensor([len(item['Y']) for item in batch]))

    def _trim_result(self, ls):
        start = 0 if len(ls) > 0 and ls[0] != self.sos_id else 1
        end = 0
        while end < len(ls) and ls[end] != self.eos_id:
            end += 1
        return ls[start:end]

    def evaluate(self, y_pred, batch, metric):
        if metric == "bleu":
            target_variables = batch['Y']
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
            return ' '.join([self.index2word[self.lang_src][word_id] for word_id in src]), \
                ' '.join([self.index2word[self.lang_tgt][word_id] for word_id in tgt]), \
                ' '.join([self.index2word[self.lang_tgt][word_id] for word_id in y_pred])
        else:
            return super().format_output(y_pred, batch_item)
