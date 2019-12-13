import csv
import json
import os

import spacy
from dlex.configs import MainConfig
from dlex.datasets import DatasetBuilder
from dlex.datasets.nlp.utils import Vocab, char_tokenize, Tokenizer, write_vocab, normalize_none, nltk_tokenize
from dlex.utils import logger
from tqdm import tqdm


def tokenize(sent: str):
    sent = sent.replace('[', ' [')
    sent = sent.replace(" '", ' "').replace("' ", '" ')
    return [t.lower() for t in nltk_tokenize(sent)]


def detokenize(sent: str):
    sent = sent.replace(' - ', '-').replace(" 's", "'s").replace(' / ', '/')
    return sent


def _is_equal(w1, w2):
    def _normalize(w):
        return w.strip().replace("''", '"').replace('``', '"').replace(' ', '')
    return _normalize(w1) == _normalize(w2)


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


nlp = spacy.blank("en")


def word_tokenize(sent):
    sent = sent.replace('.[', '. [')
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(text)
            print(tokens)
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_dataset2(working_dir, output_dir, version: str):
    total = 0
    text = []
    for mode in ['train', 'dev']:
        examples = []
        with open(os.path.join(working_dir, "%s-%s.json" % (mode, version)), encoding='utf-8') as fh:
            source = json.load(fh)

        for article in tqdm(source["data"], desc=f"Loading {mode}"):
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                spans = convert_idx(context, context_tokens)
                text += context_tokens
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace("''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    text += ques_tokens
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)

                    if (version == 'v2.0' and qa['is_impossible']) or answer_span:
                        examples.append((
                            qa['id'],
                            ' '.join(context_tokens),
                            ' '.join(ques_tokens),
                            ' '.join(['%d-%d' % (y1, y2) for y1, y2 in zip(y1s, y2s)])))

        logger.debug(f"Number of questions ({mode}): {len(examples)}")

        with open(os.path.join(output_dir, mode + '.csv'), 'w') as f:
            writer = csv.writer(f)
            for id, context, question, answer_span in examples:
                writer.writerow([id, context, question, answer_span])

    write_vocab(output_dir, text, "word.txt", min_freq=1)
    write_vocab(output_dir, text, "char.txt", Tokenizer(normalize_none, char_tokenize), min_freq=10)


def process_dataset(working_dir, output_dir, version: str):
    text = []  # to extract vocabulary
    for mode in ['train', 'dev']:
        with open(os.path.join(working_dir, "%s-%s.json" % (mode, version)), encoding='utf-8') as data_file:
            data = json.load(data_file)['data']

        examples = []
        logger.debug("Number of samples (%s): %d", mode,
                     sum([sum([len(p['qas']) for p in a['paragraphs']]) for a in data]))
        for article in tqdm(data, desc=f"Preprocessing {mode}"):
            for paragraph in article['paragraphs']:
                context = paragraph['context'].lower()
                context_tokens = tokenize(paragraph['context'])
                text.append(context_tokens)

                charloc2wordloc = get_char_word_loc_mapping(context, context_tokens)
                # charloc2wordloc maps the character location (int) of a context token to
                # a pair giving (word (string), word loc (int)) of that token
                if not charloc2wordloc:
                    logger.warning("Cannot map char to word.")
                    continue

                # for each question, process the question and answer and write to file
                for qa in paragraph['qas']:
                    question = qa['question']
                    question_tokens = tokenize(question)
                    text.append(question_tokens)

                    ans_spans = []
                    for ans in qa['answers']:
                        ans_text = ans['text'].lower()
                        ans_start_char_pos = ans['answer_start']
                        if context[ans_start_char_pos] in [" "]:
                            ans_start_char_pos += 1
                        ans_end_char_pos = ans_start_char_pos + len(ans_text)

                        # Check that the provided character spans match the provided answer text
                        if context[ans_start_char_pos:ans_end_char_pos] != ans_text:
                            #print(context)
                            #print(question)
                            #print(context[ans_start_char_pos:ans_end_char_pos], ans_text); input()
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
                        if not _is_equal(" ".join(ans_tokens), ans_text):
                            print(context_tokens)
                            print(question)
                            print(ans_tokens, ans_text);
                            input()
                            continue
                        ans_spans.append((ans_start_word_pos, ans_end_word_pos))

                    if (version == 'v2.0' and qa['is_impossible']) or ans_spans:
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

    write_vocab(output_dir, text, "word.txt")
    write_vocab(output_dir, text, "char.txt", Tokenizer(normalize_none, char_tokenize), min_freq=10)


class SQuAD(DatasetBuilder):
    """
    Builder for SQuAD dataset (v1.1 and v2.0)
        https://rajpurkar.github.io/SQuAD-explorer/
    """

    version = None

    def __init__(self, params: MainConfig):
        super().__init__(params)
        assert self.version
        self._vocab_word = None
        self._vocab_char = None

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
        for fn in ["train-%s.json" % self.version, "dev-%s.json" % self.version]:
            self.download_and_extract(base_url + fn, self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        if os.path.exists(self.get_processed_data_dir()):
            return
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)
        # process_dataset(self.get_working_dir(), self.get_processed_data_dir(), self.version)
        process_dataset2(self.get_working_dir(), self.get_processed_data_dir(), self.version)

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

    @staticmethod
    def is_better_result(metric: str, best_result: float, new_result: float):
        if metric == "em":
            return new_result > best_result
        else:
            return super(SQuAD, SQuAD).is_better_result(metric, best_result, new_result)

    def format_output(self, y_pred, batch_item) -> (str, str, str):
        return "", \
               "%s %s" % (batch_item.Y[0], batch_item.Y[1]), \
               "%s %s" % (y_pred[0], y_pred[1])


class SQuAD_v1(SQuAD):
    version = "v1.1"


class SQuAD_v2(SQuAD):
    version = "v2.0"