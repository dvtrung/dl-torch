import csv
import json
import os

import nltk
from dlex.configs import MainConfig
from dlex.datasets import DatasetBuilder
from dlex.datasets.nlp.utils import Vocab, char_tokenize, Tokenizer, write_vocab, normalize_none
from dlex.utils import logger
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def tokenize(seq):
    tokens = [t for t in nltk.word_tokenize(seq)]
    return tokens


def get_char_word_loc_mapping(context, context_tokens):
    """
    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.
    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)
    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    acc = ''  # accumulator
    current_token_idx = 0  # current word loc
    mapping = dict()

    def _is_equal(w1, w2):
        def _normalize(w):
            return w.strip().replace("''", '"').replace('``', '"')
        return _normalize(w1) == _normalize(w2)

    for char_idx, char in enumerate(context):
        if char != u' ' and char != u'\n':
            acc += char
            context_token = context_tokens[current_token_idx]  # current word token

            if _is_equal(acc, context_token):  # if the accumulator now matches the current word token
                syn_start = char_idx - len(acc) + 1  # char loc of the start of this word
                for char_loc in range(syn_start, char_idx + 1):
                    mapping[char_loc] = (acc, current_token_idx)  # add to mapping
                acc = ''  # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


def process_dataset(working_dir, output_dir):
    text = []  # to extract vocabulary
    for mode in ['train', 'dev']:
        with open(os.path.join(working_dir, "%s-v1.1.json" % mode)) as data_file:
            data = json.load(data_file)['data']

        examples = []
        logger.debug("Number of samples (%s): %d", mode,
                     sum([sum([len(p['qas']) for p in a['paragraphs']]) for a in data]))
        for article in tqdm(data, desc=f"Preprocessing {mode}"):
            for paragraph in article['paragraphs']:
                context = paragraph['context'].lower()
                context_tokens = tokenize(context)
                text.append(context_tokens)

                charloc2wordloc = get_char_word_loc_mapping(context, context_tokens)
                # charloc2wordloc maps the character location (int) of a context token to
                # a pair giving (word (string), word loc (int)) of that token
                assert charloc2wordloc

                # for each question, process the question and answer and write to file
                for qa in paragraph['qas']:
                    question = qa['question']
                    question_tokens = tokenize(question)
                    text.append(question_tokens)

                    ans_spans = []
                    for ans in qa['answers']:
                        ans_text = ans['text'].lower()
                        ans_start_char_pos = ans['answer_start']
                        ans_end_char_pos = ans_start_char_pos + len(ans_text)

                        # Check that the provided character spans match the provided answer text
                        if context[ans_start_char_pos:ans_end_char_pos] != ans_text:
                            continue

                        # get word locs for answer start and end (inclusive)
                        ans_start_word_pos = charloc2wordloc[ans_start_char_pos][1]  # answer start word loc
                        ans_end_word_pos = charloc2wordloc[ans_end_char_pos - 1][1]  # answer end word loc

                        # Check retrieved answer tokens match the provided answer text.
                        # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"
                        # and the answer character span is around "generation",
                        # but the tokenizer regards "fifth-generation" as a single token.
                        # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                        ans_tokens = context_tokens[ans_start_word_pos:ans_end_word_pos + 1]
                        if "".join(ans_tokens) != "".join(ans_text.split()):
                            continue
                        ans_spans.append((ans_start_word_pos, ans_end_word_pos))

                    if ans_spans:
                        examples.append((
                            qa['id'],
                            ' '.join(context_tokens),
                            ' '.join(question_tokens),
                            ' '.join(['%d-%d' % (start, end) for start, end in ans_spans])))

        logger.debug("Number of processed samples (%s): %d", mode, len(examples))

        with open(os.path.join(output_dir, mode + '.csv'), 'w') as f:
            writer = csv.writer(f)
            for id, context, question, answer_span in examples:
                writer.writerow([id, context, question, answer_span])

    write_vocab(output_dir, text, "word", min_freq=1)
    write_vocab(output_dir, text, "char", Tokenizer(normalize_none, char_tokenize), min_freq=20)


class SQuAD_V1(DatasetBuilder):
    """
    Builder for SQuAD dataset (v1.0)
        https://rajpurkar.github.io/SQuAD-explorer/
    """

    def __init__(self, params: MainConfig):
        super().__init__(params)

        self._vocab_word = None
        self._vocab_char = None

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
        for fn in ["train-v1.1.json", "dev-v1.1.json"]:
            self.download_and_extract(base_url + fn, self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        if os.path.exists(self.get_processed_data_dir()):
            return
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)
        process_dataset(self.get_working_dir(), self.get_processed_data_dir())

    def get_vocab_path(self, tag):
        return os.path.join(self.get_processed_data_dir(), "vocab", "%s.txt" % tag)

    @property
    def vocab_word(self) -> Vocab:
        if self._vocab_word is None:
            self._vocab_word = Vocab.from_file(self.get_vocab_path("word"))
        return self._vocab_word

    @property
    def vocab_char(self) -> Vocab:
        if self._vocab_char is None:
            self._vocab_char = Vocab.from_file(self.get_vocab_path("char"))
        return self._vocab_char

    def get_pytorch_wrapper(self, mode: str):
        from .torch import PytorchSQuAD_V1
        return PytorchSQuAD_V1(self, mode)

    def evaluate(self, pred, ref, metric: str):
        # Evaluation is handled inside the dataset instance
        raise NotImplementedError

    def format_output(self, y_pred, batch_item) -> (str, str, str):
        return "", \
               "%s %s" % (batch_item.Y[0], batch_item.Y[1]), \
               "%s %s" % (y_pred[0], y_pred[1])