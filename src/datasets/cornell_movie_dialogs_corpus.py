"""
Download URL: http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
"""
import os
import codecs
import csv

from datasets.base.nlp import NLPDataset
from utils.utils import maybe_download, maybe_unzip


class Corpus(NLPDataset):
    working_dir = os.path.join("datasets", "cornell_movie_dialogs_corpus")
    raw_data_dir = os.path.join(working_dir, "raw", "cornell movie-dialogs corpus")
    processed_data_dir = os.path.join(working_dir, "data")

    def __init__(self):
        super(NLPDataset).__init__()

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        download_url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
        maybe_download(
            "data.zip",
            cls.working_dir,
            download_url)
        maybe_unzip("data.zip", cls.working_dir, "raw")

    @classmethod
    def maybe_preprocess(cls, force=False):
        if not force and os.path.exists(cls.processed_data_dir):
            return

        os.makedirs(cls.processed_data_dir, exist_ok=True)

        # Splits each line of the file into a dictionary of fields
        def loadLines(fileName, fields):
            lines = {}
            with open(fileName, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    values = line.split(" +++$+++ ")
                    # Extract fields
                    lineObj = {}
                    for i, field in enumerate(fields):
                        lineObj[field] = values[i]
                    lines[lineObj['lineID']] = lineObj
            return lines

        # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
        def loadConversations(fileName, lines, fields):
            conversations = []
            with open(fileName, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    values = line.split(" +++$+++ ")
                    # Extract fields
                    convObj = {}
                    for i, field in enumerate(fields):
                        convObj[field] = values[i]
                    # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                    lineIds = eval(convObj["utteranceIDs"])
                    # Reassemble lines
                    convObj["lines"] = []
                    for lineId in lineIds:
                        convObj["lines"].append(lines[lineId])
                    conversations.append(convObj)
            return conversations

        # Extracts pairs of sentences from conversations
        def extractSentencePairs(conversations):
            qa_pairs = []
            for conversation in conversations:
                # Iterate over all the lines of the conversation
                for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                    input_line = conversation["lines"][i]["text"].strip()
                    target_line = conversation["lines"][i + 1]["text"].strip()
                    # Filter wrong samples (if one of the lists is empty)
                    if input_line and target_line:
                        qa_pairs.append([input_line, target_line])
            return qa_pairs

        # Define path to new file
        datafile = os.path.join(cls.processed_data_dir, "formatted_movie_lines.txt")

        delimiter = '\t'
        # Unescape the delimiter
        delimiter = str(codecs.decode(delimiter, "unicode_escape"))

        # Initialize lines dict, conversations list, and field ids
        movie_lines_fields = ["lineID", "characterID", "movieID", "character", "text"]
        movie_conversations_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

        # Load lines and process conversations
        print("\nProcessing corpus...")
        lines = loadLines(os.path.join(cls.raw_data_dir, "movie_lines.txt"), movie_lines_fields)
        print("\nLoading conversations...")
        conversations = loadConversations(os.path.join(cls.raw_data_dir, "movie_conversations.txt"),
                                          lines, movie_conversations_fields)

        # Write new csv file
        print("\nWriting newly formatted file...")
        with open(datafile, 'w', encoding='utf-8') as output_file:
            writer = csv.writer(output_file, delimiter=delimiter, lineterminator='\n')
            for pair in extractSentencePairs(conversations):
                writer.writerow(pair)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass