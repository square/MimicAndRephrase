import csv
from typing import NamedTuple, Sequence, List, Tuple, Set

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils.word_embedding import Glove

import numpy as np

class DataPoint(NamedTuple):
    """
    Generate a named tuple class to make sure that DataPoint is also immutable
    """
    raw_question: List[str]
    raw_response: List[str]
    question_indices: Tensor
    response_indices: Tensor
    copy_indices: Tensor
    sentiment_flags: Tensor

class IDKBatch(NamedTuple):
    raw_questions: List[List[str]]
    raw_responses: List[List[str]]
    question_indices: Tensor
    question_lengths: Tensor
    response_indices: Tensor
    response_lengths: Tensor
    copy_indices: Tensor
    sentiment_flags: Tensor

class IDKRephraseDataset(Dataset):
    def __init__(self, data: List[DataPoint]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def from_TSV(cls, file_name: str, glove: Glove, sos: str="<SOS>", eos: str="<EOS>", response_vocab: List[str]=None,
                 use_sentiment: bool=False):
        with open(file_name, 'r') as f:
            csvreader = csv.reader(f, delimiter="\t")
            data = []
            for row in csvreader:
                if use_sentiment:
                    question = sos + " " + row[0] + " " + eos
                else:
                    question = row[0]
                response = row[1] + " " + eos
                question_indices = glove.words_to_indices(question.split())
                response_indices_glove = glove.words_to_indices(response.split())
                copy_indices = IDKRephraseDataset.get_copy_indices(question.split(), response.split())
                if response_vocab is None:
                    response_indices = response_indices_glove
                else:
                    words = response.split()
                    response_indices = [response_vocab.index(token)
                                        if token in response_vocab else len(response_vocab) for token in words]
                sentiment_flags = [0, 0]
                if use_sentiment:
                    sentiment_flags = [row[2] == "pos", row[3] == "condensed"]

                data.append(DataPoint(question.split(), response.split(), torch.LongTensor(question_indices),
                                      torch.LongTensor(response_indices), torch.Tensor(copy_indices),
                                      torch.FloatTensor(sentiment_flags)))
            return IDKRephraseDataset(data)

    @classmethod
    def get_copy_indices(cls, question_tokens: List[str], response_tokens: List[str]):
        copy_matrix = np.zeros((len(question_tokens), len(response_tokens)), dtype=np.int32)
        for i in range(len(question_tokens)):
            for j in range(len(response_tokens)):
                if question_tokens[i].lower() == response_tokens[j].lower():
                    copy_matrix[i, j] = 1
        return copy_matrix

    @classmethod
    def get_vocab_from_TSV(cls, file_name: str) -> Set[str]:
        vocab = set()
        with open(file_name, 'r') as f:
            csvreader = csv.reader(f, delimiter="\t")
            for row in csvreader:
                for column in row:
                    vocab |= set(column.split())

        return vocab

class BatchCollate:
    """
    A collate object (so that it is pickle-able) for PyTorch's DataLoader
    """
    def __call__(self, data_points: List[DataPoint]) -> IDKBatch:
        raw_questions = [point.raw_question for point in data_points]
        raw_responses = [point.raw_response for point in data_points]
        question_lengths = np.asarray([len(point.question_indices) for point in data_points], dtype=np.int32)
        response_lengths = np.asarray([len(point.response_indices) for point in data_points], dtype=np.int32)
        question_indices = np.asarray([BatchCollate.pad_array(point.question_indices, max(question_lengths))
                                       for point in data_points], dtype=np.int64)
        response_indices = np.asarray([BatchCollate.pad_array(point.response_indices, max(response_lengths))
                                       for point in data_points], dtype=np.int64)
        copy_indices = np.asarray([BatchCollate.pad_array(
            BatchCollate.pad_array(point.copy_indices, max(question_lengths)), max(response_lengths), axis=1)
            for point in data_points], dtype=np.int64)
        return IDKBatch(raw_questions, raw_responses,
                        torch.LongTensor(question_indices), torch.LongTensor(question_lengths),
                        torch.LongTensor(response_indices), torch.LongTensor(response_lengths),
                        torch.Tensor(copy_indices), torch.stack([point.sentiment_flags for point in data_points]))

    @staticmethod
    def pad_array(arr: np.ndarray, length: int, axis: int=0):
        """
        :param arr: Numpy array to be padded
        :param length: Final length of the padded array
        :param axis: Axis along which array will be padded
        :return:
        """
        if arr.shape[axis] >= length:
            return np.asarray(arr, dtype=np.int64)
        else:
            zeros_shape = list(arr.shape)
            zeros_shape[axis] = length - arr.shape[axis]
            zeros_shape = tuple(zeros_shape)
            zeros = np.zeros(zeros_shape, dtype=np.int64)
            padded = np.concatenate([arr, zeros], axis=axis)
            return padded


def default_data_loader(dataset: Dataset, batch_size: int, num_workers: int=0, shuffle: bool=True) -> DataLoader:
    return DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=BatchCollate(), num_workers=num_workers)
