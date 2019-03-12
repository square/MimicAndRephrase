from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from utils.maskedsequence import MaskedSequence
from utils import functions as f
from utils.utils import get_device


class LearnedAttention(nn.Module):
    """
    Performs attention on a single MaskedSequence by learning a weight vector, computing logits per time step, softmax,
    and then a sum reweight.
    """

    def __init__(self, input_dim: int, hidden_dim: int, attn_dim: int = 512, extra_layer: bool = False,
                 extra_layer_size: int = 512, dropout: float = 0.5, use_coverage: bool = False, use_bias = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attnv = nn.Linear(attn_dim, 1, bias=False)
        self.extra_layer = extra_layer
        self.use_coverage = use_coverage
        coverage_bit = 0
        if use_coverage:
            coverage_bit = 1
        if extra_layer:
            self.dropout = dropout
            self.linear = nn.Linear(extra_layer_size, attn_dim, bias=False)
            self.relu = nn.ReLU()
            self.first_layer = nn.Linear(input_dim + hidden_dim + coverage_bit, extra_layer_size, bias=True)
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.linear = nn.Linear(input_dim + hidden_dim + coverage_bit, attn_dim, bias=use_bias)

    def forward(self, sequence: MaskedSequence, hidden: Tensor, coverage: Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_var = sequence.as_masked()
        exhidden = torch.unsqueeze(hidden, 1).expand((sequence_var.size()[0], sequence_var.size()[1], hidden.size()[1]))
        if self.use_coverage and coverage is None:
            coverage = exhidden.new_full((sequence_var.size()[0], sequence_var.size()[1], 1), 0)
        if self.use_coverage:
            copy_in = torch.cat((sequence_var, exhidden, coverage), dim=2)
        else:
            copy_in = torch.cat((sequence_var, exhidden), dim=2)
        if self.extra_layer:
            copy_in = self.relu(self.first_layer(copy_in))

        logits = self.attnv(torch.tanh(self.linear(copy_in)))  # (N, L, 1)
        # todo correct mask generation so that it uses internal mask
        exp_mask = f.to_exp_mask(
            MaskedSequence.gen_mask_from_shape(
                [*logits.size()[:2], 1],
                sequence.lengths,
                get_device(sequence_var),
            )
        )  # (N, L, 1)
        logits_exp_masked = logits + exp_mask  # (N, L, 1)
        scores = F.softmax(logits_exp_masked, dim=-2)
        coverage_loss = None
        new_coverage = None
        if self.use_coverage:
            coverage_loss = torch.sum(torch.min(coverage, scores))
            new_coverage = coverage + scores
        return torch.squeeze(torch.matmul(torch.transpose(scores, -1, -2), sequence_var), 1), logits_exp_masked, \
               new_coverage, coverage_loss
