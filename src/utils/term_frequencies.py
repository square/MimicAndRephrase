from log import info, track
import csv
from collections import Counter
from typing import Mapping, Sequence, List
import os
from core_nlp import SimpleSentence


class TermFrequencies(Mapping[str, int]):
    """
    An object that helps with all things related to the counting  / retrieving frequencies of tokens
    """
    def __init__(self, count_mapping: Mapping[str, int], total_count: int):
        self.total_count = total_count
        self._vocab_counts: Counter = Counter(count_mapping)

    @classmethod
    def from_file(cls, vocab_counts_file: str=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../resources/google_unigram_counts.tsv")) -> 'TermFrequencies':
        """
        :param vocab_counts_file: Path to the vocab count tsv. Format is {token}\t{count}.
        :return: A TermFrequencies object
        """
        with track("Reading unigram counts"):
            vocab_counts = Counter()
            idx = 0
            with open(vocab_counts_file, 'r', encoding='utf-8') as f:
                csvreader = csv.reader(f, delimiter='\t')
                total_count = int(next(csvreader)[1])
                for row in csvreader:
                    idx += 1
                    if idx % 100000 == 0:
                        info("Processed {} tokens.".format(idx))
                    word = row[0]
                    count = int(row[1])
                    vocab_counts[word] = count
        return TermFrequencies(vocab_counts, total_count)

    @classmethod
    def from_sentences(cls, sentences: Sequence[SimpleSentence]) -> 'TermFrequencies':
        """
        Loads a Term Frequencies object by counting tokens in a list of sentences
        :param sentences: A list of sentences
        """
        count = Counter()
        total = 0
        for sentence in sentences:
            for token in sentence.original_texts():
                count[token] += 1
                total += 1

        return TermFrequencies(count, total)

    def most_common(self, n: int) -> List[str]:
        """
        Same behavior as Counter.most_common()
        :param n:
        :return: List of the n most common elements and their counts from the most common to the least.
        """
        return self._vocab_counts.most_common(n)

    def __getitem__(self, item):
        return self._vocab_counts[item]

    def __iter__(self):
        return self._vocab_counts.__iter__()

    def __len__(self):
        return self._vocab_counts.__len__()

    def __contains__(self, o):
        return self._vocab_counts.__contains__(o)

    def keys(self):
        return self._vocab_counts.keys()

    def values(self):
        return self._vocab_counts.values()

    def __eq__(self, other: 'TermFrequencies'):
        return self._vocab_counts.__eq__(other._vocab_counts)

    def __ne__(self, other: 'TermFrequencies'):
        return self._vocab_counts.__ne__(other._vocab_counts)

    def __add__(self, other: 'TermFrequencies'):
        return self._vocab_counts.__add__(other._vocab_counts)

    @classmethod
    def mock(cls):
        return cls(Counter(["the", "a", "an"]), 3)
