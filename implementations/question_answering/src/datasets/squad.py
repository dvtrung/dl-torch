import json
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Dict

import nltk
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.nlp.utils import write_vocab, normalize_lower, Vocab, char_tokenize, Tokenizer
from dlex.datasets.torch import Dataset
from dlex.torch import Batch, BatchItem
from dlex.torch.utils.ops_utils import LongTensor
from dlex.utils import logger

BatchY = namedtuple("BatchY", "answer_span")


class QABatch(Batch):
    @dataclass
    class BatchX:
        context_word: torch.Tensor
        context_char: torch.Tensor
        question_word: torch.Tensor
        question_char: torch.Tensor

    X: BatchX
    Y: BatchY
    X_len: BatchX
    Y_len: BatchY

    def __len__(self):
        return len(self.X.context_word)

    def item(self, i: int) -> BatchItem:
        return BatchItem(
            X=None,
            Y=self.Y[i].cpu().detach().numpy())


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
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

    for char_idx, char in enumerate(context):
        if char != u' ' and char != u'\n':
            acc += char
            context_token = context_tokens[current_token_idx]  # current word token
            if acc == context_token:  # if the accumulator now matches the current word token
                syn_start = char_idx - len(acc) + 1  # char loc of the start of this word
                for char_loc in range(syn_start, char_idx + 1):
                    mapping[char_loc] = (acc, current_token_idx)  # add to mapping
                acc = ''  # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


class SQuAD_V1(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        for fn in ["train-v1.1.json", "dev-v1.1.json"]:
            self.download_and_extract(
                "https://rajpurkar.github.io/SQuAD-explorer/dataset/" + fn,
                self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        if os.path.exists(self.get_processed_data_dir()):
            return
        os.makedirs(self.get_processed_data_dir(), exist_ok=True)

        for mode in ['train', 'dev']:
            with open(os.path.join(self.get_working_dir(), "%s-v1.1.json" % mode)) as data_file:
                data = json.load(data_file)

            logger.info(
                "%s size: %d", mode,
                sum([sum([len(para['qas']) for para in article['paragraphs']]) for article in data['data']]))

            num_exs = 0  # number of examples written to file
            num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0
            examples = []

            for articles_id in tqdm(range(len(data['data'])), desc=f"Preprocessing {mode}"):
                article_paragraphs = data['data'][articles_id]['paragraphs']
                for pid in range(len(article_paragraphs)):
                    context = article_paragraphs[pid]['context']
                    # The following replacements are suggested in the paper
                    # BidAF (Seo et al., 2016)
                    context = context.replace("''", '" ')
                    context = context.replace("``", '" ')

                    context = context.lower()
                    context_tokens = tokenize(context)

                    qas = article_paragraphs[pid]['qas']  # list of questions

                    charloc2wordloc = get_char_word_loc_mapping(context, context_tokens)
                    # charloc2wordloc maps the character location (int) of a context token to a pair giving (word (string), word loc (int)) of that token
                    if charloc2wordloc is None:  # there was a problem
                        num_mappingprob += len(qas)
                        continue  # skip this context example

                    # for each question, process the question and answer and write to file
                    for qn in qas:
                        question = qn['question']
                        question_tokens = tokenize(question)

                        ans_text = qn['answers'][0]['text'].lower()  # get the answer text
                        ans_start_charloc = qn['answers'][0]['answer_start']  # answer start loc (character count)
                        ans_end_charloc = ans_start_charloc + len(
                            ans_text)  # answer end loc (character count) (exclusive)

                        # Check that the provided character spans match the provided answer text
                        if context[ans_start_charloc:ans_end_charloc] != ans_text:
                            # Sometimes this is misaligned, mostly because "narrow builds" of Python 2 interpret certain Unicode characters to have length 2 https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string
                            # We should upgrade to Python 3 next year!
                            num_spanalignprob += 1
                            continue

                        # get word locs for answer start and end (inclusive)
                        ans_start_wordloc = charloc2wordloc[ans_start_charloc][1]  # answer start word loc
                        ans_end_wordloc = charloc2wordloc[ans_end_charloc - 1][1]  # answer end word loc
                        assert ans_start_wordloc <= ans_end_wordloc

                        # Check retrieved answer tokens match the provided answer text.
                        # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"
                        # and the answer character span is around "generation",
                        # but the tokenizer regards "fifth-generation" as a single token.
                        # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                        ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc + 1]
                        if "".join(ans_tokens) != "".join(ans_text.split()):
                            num_tokenprob += 1
                            continue  # skip this question/answer pair

                        examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(ans_tokens),
                                         ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))

                        num_exs += 1

            logger.info(
                "Number of (context, question, answer) triples discarded due to char -> token mapping problems: %d",
                num_mappingprob)
            logger.info(
                "Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: %d",
                num_tokenprob)
            logger.info(
                "Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): %d",
                num_spanalignprob)
            logger.info("Processed %d examples of total %d\n" % (
            num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))

            if mode == "train":
                write_vocab(
                    self.get_processed_data_dir(),
                    [s for ex in examples for s in ex[:3]],
                    output_file_name="word.txt",
                    normalize_fn=normalize_lower,
                    tokenize_fn=tokenize,
                    min_freq=2)
                write_vocab(
                    self.get_processed_data_dir(),
                    [s for ex in examples for s in ex[:3]],
                    output_file_name="char.txt",
                    normalize_fn=normalize_lower,
                    tokenize_fn=char_tokenize,
                    min_freq=5)

            with open(os.path.join(self.get_processed_data_dir(), mode + '.csv'), 'w') as f:
                f.write('\n'.join([
                    f"{context}\t{question}\t{answer}\t{answer_span}"
                    for context, question, answer, answer_span in examples]))

    def get_pytorch_wrapper(self, mode: str):
        return PytorchSQuAD_V1(self, mode)

    def evaluate(self, pred, ref, metric: str):
        if metric == "acc":
            return accuracy_score(
                ["%s-%s" % (r[0], r[1]) for r in ref],
                ["%s-%s" % (p[0], p[1]) for p in pred])

    def format_output(self, y_pred, batch_item) -> (str, str, str):
        return "", \
               "%s %s" % (batch_item.Y[0], batch_item.Y[1]), \
               "%s %s" % (y_pred[0], y_pred[1])


class PytorchQADataset(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
        self.vocab_word = Vocab(
            os.path.join(builder.get_processed_data_dir(), "vocab", "word.txt"),
            Tokenizer(normalize_lower, tokenize))
        self.vocab_char = Vocab(
            os.path.join(builder.get_processed_data_dir(), "vocab", "char.txt"),
            Tokenizer(normalize_lower, char_tokenize))

    @property
    def vocab_size_word(self):
        return len(self.vocab_word)

    @property
    def vocab_size_char(self):
        return len(self.vocab_char)


class PytorchSQuAD_V1(PytorchQADataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
        data = []
        with open(os.path.join(builder.get_processed_data_dir(), mode + ".csv")) as f:
            for line in tqdm(f.read().split("\n"), desc="Loading data (%s)" % mode):
                line = line.split('\t')
                context = self.vocab_word.tokenize(line[0])
                if len(context) > self.params.dataset.paragraph_max_length:
                    continue
                question = self.vocab_word.tokenize(line[1])
                if len(question) > self.params.dataset.question_max_length:
                    continue
                data.append(dict(
                    cw=self.vocab_word.encode_token_list(context),
                    qw=self.vocab_word.encode_token_list(question),
                    cc=[self.vocab_char.encode_str(w) for w in context],
                    qc=[self.vocab_char.encode_str(w) for w in question],
                    answer=self.vocab_word.encode_str(line[2]),
                    answer_span=[int(pos) for pos in line[3].split(' ')]
                ))
        self._data = data

    def collate_fn(self, batch: List[Dict]):
        # batch.sort(key=lambda item: len(item.X), reverse=True)
        w_contexts = [LongTensor(item['cw']) for item in batch]
        w_questions = [LongTensor(item['qw']) for item in batch]

        char_max_length = max([max(len(c) for c in item['cc']) for item in batch])
        c_contexts = [LongTensor([
            char_idx + (char_max_length - len(char_idx)) * [self.vocab_char.blank_token_idx]
            for char_idx in item['cc']
        ]) for item in batch]

        char_max_length = max([max(len(c) for c in item['qc']) for item in batch])
        c_questions = [LongTensor([
            char_idx + (char_max_length - len(char_idx)) * [self.vocab_char.blank_token_idx]
            for char_idx in item['qc']
        ]) for item in batch]

        # answers = [torch.LongTensor(item[2]) for item in batch]
        answer_spans = LongTensor([item['answer_span'] for item in batch])

        context_word_lengths = [len(c) for c in w_contexts]
        question_word_lengths = [len(q) for q in w_questions]

        w_contexts = nn.utils.rnn.pad_sequence(
            w_contexts, batch_first=True, padding_value=self.vocab_word.blank_token_idx)
        w_questions = nn.utils.rnn.pad_sequence(
            w_questions, batch_first=True, padding_value=self.vocab_word.blank_token_idx)
        c_contexts = nn.utils.rnn.pad_sequence(
            c_contexts, batch_first=True, padding_value=self.vocab_char.blank_token_idx)
        c_questions = nn.utils.rnn.pad_sequence(
            c_questions, batch_first=True, padding_value=self.vocab_char.blank_token_idx)
        # answers = nn.utils.rnn.pad_sequence(answers, batch_first=True, padding_value=self.vocab.blank_token_idx)

        return QABatch(
            X=QABatch.BatchX(w_contexts, c_contexts, w_questions, c_questions),
            X_len=QABatch.BatchX(context_word_lengths, None, question_word_lengths, None),
            Y=answer_spans, Y_len=None)
