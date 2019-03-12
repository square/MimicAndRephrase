from typing import Sequence, List, Tuple, Type, Union, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

import functions as mytorch
from utils import get_device

class MaskedSequence:
    """
    A helpful class for handling masked sequences like batches of sentences embeddings. Basically, you can wrap
    an unmasked tensor, a list of lengths corresponding to the lengths of each sequence in the batch, into this object.
    """
    def __init__(self, sequence: torch.Tensor, lengths: List[int], is_pre_masked=False, mask=None):
        """
        :param sequence: (batch_size, length, embedding dimension)
        :param lengths: list of lengths each corresponding to the length of the respective sequence
        """
        self._data: torch.Tensor = sequence
        self.lengths: List[int] = lengths
        self.order: torch.Tensor = None
        self.revert_order: torch.Tensor = None
        self._mask: Optional[torch.Tensor] = mask
        self._masked_data: Optional[torch.Tensor] = self._data if is_pre_masked else None
        self.packed_sequence: Optional[PackedSequence] = None
        self._lengths_for_packed_sequences: List[int] = None

    def as_masked(self) -> torch.Tensor:
        """
        Get a masked version of the data. This output is only computed the first time it is called, subsequent calls
        return the cached result.
        :return:
        """
        if self._masked_data is not None:
            return self._masked_data
        else:
            mask = self.get_mask()
            self._masked_data = self._data * mask
        return self._masked_data

    def get_data(self) -> torch.Tensor:
        """
        Gets the sequence tensor that was originally stored when constructing this MaskedSequence. Masking has not
        been applied to this tensor.
        """
        return self._data

    def get_mask(self) -> torch.Tensor:
        """
        Gets the mask for this sequence. This is computed lazily.
        The mask will always be shaped: (batch_size, max_len, 1). The final one is very handy for commonly
        used broadcasts (like masking things...)
        :return:
        """
        if self._mask is None:
            self._mask = MaskedSequence.gen_mask(self._data, self.lengths).type_as(self._data)
        return self._mask

    def as_packed_sequence(self) -> PackedSequence:
        """
        Get the packed_sequence version of this MaskedSequence. Remember, packed_sequences have been sorted into
        order of descending lengths
        :return:
        """
        if self.packed_sequence is None:
            self.packed_sequence = pack_padded_sequence(
                self.sort(self._data),
                # This adjustment is because PackedSequences do not allow 0 length sequences
                sorted(self.__get_lengths_for_packed_sequences(), reverse=True),
                batch_first=True
            )
        return self.packed_sequence

    def from_padded_sequence(self, padded_sequence: torch.Tensor, is_pre_masked=False) -> 'MaskedSequence':
        return self.with_new_data(self.unsort(padded_sequence), is_pre_masked=is_pre_masked)

    def from_packed_sequence(self, packed_sequence: PackedSequence) -> 'MaskedSequence':
        """
        Convert a packed sequence into a masked sequence using this MaskedSequence's length and mask. This is useful
        when a PackedSequence you got from this MaskedSequence was fed into a PyTorch RNN and you would like to convert
        the resulting PackedSequence back into a MaskedSequence (performing unsorting for example and using the same
        mask / lengths).
        :param packed_sequence: The packed sequence you would like to convert to a MaskedSequence using this
                                MaskedSequence's lengths and mask.
        :return: A MaskedSequence
        """
        padded_sequence = pad_packed_sequence(packed_sequence, batch_first=True)
        return self.with_new_data(
            self.unsort(padded_sequence),
            is_pre_masked=False  # this is false in case we have 0 length sequences disguised as 1 length sequences
        )

    def with_new_data(self, new_data: torch.Tensor, is_pre_masked: bool=False):
        """
        Creates a new MaskedSequence with the same lengths and same mask, but with a new data tensor. This new data
        tensor must be the same shape as the original data in the first two dimensions (batch size and length) but can
        have a different embedding dimension size.
        :param new_data: The new data tensor
        :param is_pre_masked: Whether this new tensor is already masked
        :return:
        """
        return MaskedSequence(
            new_data,
            self.lengths,
            is_pre_masked=is_pre_masked,
            mask=self._mask,
        )

    def sort(self, var: torch.Tensor) -> torch.Tensor:
        """
        Sorts var in the same way that data needs to be sorted in order to have lengths be decreasing
        :param var: (batch_size, *)
        :return:
        """
        if self.order is None:
            lengths_array = np.asarray(self.__get_lengths_for_packed_sequences())
            self.order = torch.LongTensor(np.argsort(-lengths_array))
        if self._data.is_cuda:
            self.order = self.order.cuda()
        return torch.index_select(var, 0, self.order)

    def unsort(self, var: torch.Tensor) -> torch.Tensor:
        """
        Given a variable that has been sorted by this MaskedSequence, unsort it. This is super useful
        when converting from an output that is a packed sequence. For example, MaskedSequence "foo" is converted
        to a PackedSequence and that PackedSequence is fed through and LSTM which outputs another PackedSequence, "bar".
        Now PackedSequence "bar" can be unsorted so that we can create another MaskedSequence out of it if desired.
        :param var:
        :return:
        """
        if self.revert_order is None:
            lengths_array = np.asarray(self.__get_lengths_for_packed_sequences())
            self.revert_order = torch.LongTensor(np.argsort(np.argsort(-lengths_array)))
        if self._data.data.is_cuda:
            self.revert_order = self.revert_order.cuda()
        return torch.index_select(var, 0, self.revert_order)

    def is_masked(self) -> bool:
        """
        Returns whether or not the masking computation has already been completed, this helps if you want
        to make optimizations to reduce the number of times the mask is applied to the minimum possible
        :return:
        """
        return self._masked_data is not None

    @staticmethod
    def gen_mask(var: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        """
        Generates a mask for a variable and a list of lengths. Tensor (batch size * len * d).
        :return: A torch.ByteTensor with 1s and 0s.
        """
        return MaskedSequence.gen_mask_from_shape(
            var.size(),
            lengths,
            get_device(var),
        )

    @staticmethod
    def gen_mask_from_shape(shape: Sequence[int], lengths: List[int], device: int=-1) -> torch.Tensor:
        batch_size = shape[0]
        max_length = shape[1]
        lengths_tensor = torch.Tensor(lengths)
        if device >= 0:
            lengths_tensor = lengths_tensor.cuda(device)
        lengths_tensor = torch.unsqueeze(lengths_tensor, -1)

        if device >= 0 and batch_size >= 32:
            mask = torch.gt(lengths_tensor, torch.arange(0, max_length).cuda(device))
            mask = torch.unsqueeze(mask, -1)
            return mask
        else:
            mask = torch.ByteTensor(batch_size, max_length)
            if device >= 0:
                mask = mask.cuda(device)
            mask.fill_(1)
            for i, length in enumerate(lengths):
                if length < max_length:
                    mask[i, length:] = 0
            return torch.unsqueeze(mask, -1)

    def cuda(self, device: int=None, non_blocking=False):
        new_seq = MaskedSequence(
            self.get_data().contiguous().cuda(device=device, non_blocking=non_blocking),
            self.lengths,
            is_pre_masked=False,
            mask=self._mask.contiguous().cuda(device=device, non_blocking=non_blocking) if self._mask is not None else None,
        )
        new_seq._masked_data = self._masked_data.contiguous().cuda(device=device, non_blocking=non_blocking) \
            if self._masked_data is not None else None
        return new_seq

    def cpu(self):
        new_seq = MaskedSequence(
            self.get_data().cpu(),
            self.lengths,
            is_pre_masked=False,
            mask=self._mask.cpu() if self._mask is not None else None,
        )
        new_seq._masked_data = self._masked_data.cpu() if self._masked_data is not None else None
        return new_seq

    @classmethod
    def from_sequences(cls, sequences: Sequence['MaskedSequence']) -> 'MaskedSequence':
        """
        Joins multiple sequences into 1. Takes care of padding!
        :param sequences: The masked sequences to join
        :return:
        """
        internal_vars = [seq.get_data() for seq in sequences]
        new_internal_var = torch.cat(mytorch.pad_tensors(internal_vars, dims=(1,)), dim=0)
        return MaskedSequence(new_internal_var, [length for seq in sequences for length in seq.lengths])

    def __get_lengths_for_packed_sequences(self):
        if self._lengths_for_packed_sequences is None:
            self._lengths_for_packed_sequences = [length if length > 0 else 1 for length in self.lengths]
        return self._lengths_for_packed_sequences
