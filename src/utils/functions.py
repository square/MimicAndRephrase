import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Sequence, List, Optional
from utils import LARGE_NEGATIVE, get_device


def to_exp_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Converts a mask of 1s and 0s where 1 is an element to keep and 0 is an element to mask out into a mask
    used for exponentiation. Thus, the 1s in the original mask become 0s and the 0s become a very large negative number.

    Target tensor can be masked with the output exp mask by addition.

    Example::
        >>> mask = torch.Tensor([1.0, 1.0, 0.0])
        >>> to_softmax = torch.Tensor([1.0, 2.0, 3.0])
        >>> exp_mask = to_exp_mask(mask)
        >>> probabilities = F.softmax(to_softmax + exp_mask)

    """
    return (1 - mask.float()) * LARGE_NEGATIVE


def compute_binary_accuracy(
        prediction: torch.Tensor,
        gold: torch.Tensor,
) -> float:
    """
    Get the accuracy from a set of predictions and gold labels
    :param prediction: (N,) Torch variable where each element is probability of True / 1 label
    :param gold: (N,) Same shape as prediction
    :return:
    """
    correct = torch.eq(torch.gt(prediction, 0.5).float(), gold).float().sum().data.cpu().numpy()
    return float(correct) / gold.size()[0]


def relu_init(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor], num_inputs: int) -> None:
    """
    Implementation based on http://cs231n.github.io/neural-networks-2/#init which references
    He et al. https://arxiv.org/pdf/1502.01852.pdf
    :param tensor: Tensor to be initialized
    :param num_inputs: The number of inputs in the layer
    """
    nn.init.normal_(tensor, mean=0, std=np.sqrt(2.0 / num_inputs))


def angle_dist(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    """
    Combines vectors a and by concatenating [a * b, |a-b|]
    :param vec_a: A vector
    :param vec_b: Another vector with the same shape as vec_a
    :return: The combined vector
    """
    angle_vec = vec_a * vec_b
    dist_vec = torch.abs(vec_a - vec_b)
    return torch.cat((angle_vec, dist_vec), -1)


def dropout_sequence(input: torch.Tensor, p: float, training: bool, batch_first=True):
    """
    Apply's dropout for a Batch size * sequence length * num dimensions sized variable where the same
    dropout mask is applied to each timestep
    :param input: The input variable. Must have 3 dimensions
    :param p: Probability that value will be dropped out
    :param training: If True, dropout is in training mode
    :return:
    """
    num_dims = len(input.size())
    assert num_dims == 3, "torch.Tensor must be batch size * length * num dimensions"
    if batch_first:
        expanded = torch.unsqueeze(torch.transpose(input, 1, 2), -1)
        expanded_dropped = F.dropout2d(expanded, p=p, training=training)
        return torch.transpose(torch.squeeze(expanded_dropped, -1), 1, 2)
    else:
        expanded = torch.unsqueeze(input.permute(1, 2, 0), -1)
        expanded_dropped = F.dropout2d(expanded, p=p, training=training)
        return torch.squeeze(expanded_dropped, -1).permute(2, 0, 1)


def pad_tensors(
        tensors: Union[Sequence[torch.Tensor], Sequence[torch.Tensor]],
        dims: Sequence[int]=(-1,),
        value=0,
) -> List[torch.Tensor]:
    """
    Pads a sequence of tensors that share the same shape other than their last dimension.
    :param tensors: Sequence of tensors that share the same shape other than their last dimension
    :param dims: A list of dimensions to pad to the same size
    :param value: The value to pad with.
    :return: The padded tensor
    """
    if len(tensors) == 0:
        return []
    else:
        max_lens = [-1 for _ in range(len(tensors[0].size()))]  # [-1, ..., -1]
        for tensor in tensors:
            for i in dims:
                if max_lens[i] < tensor.size()[i]:
                    max_lens[i] = tensor.size()[i]

        padded_tensors = []
        padding = [0 for _ in range(2 * len(max_lens))]
        for tensor in tensors:
            for dim in dims:
                #  weird indexing is just to place the number in the right position, since F.pad takes the padding sizes
                #  in reverse order
                padding[(dim * 2) * -1 - 1] = max_lens[dim] - tensor.size()[dim]
            padded_tensors.append(F.pad(tensor, tuple(padding), value=value))
        return padded_tensors


def pad_and_stack_tensors(
        tensors: Sequence[Optional[torch.Tensor]],
        value=0,
) -> torch.Tensor:
    """
    pads and stacks a list of tensors. The list of tensors can be any size (but same # of dimensions) and it will
    try and stack them by padding all of them to be the size of the largest tensor.
    """
    # Check to see that they all have same number of dimensions
    prev_size = None
    for tensor in tensors:
        if tensor is not None and prev_size is not None and len(tensor.size()) != prev_size:
            raise ValueError("Tensors must have the same number of dimensions")
        elif tensor is not None:
            prev_size = len(tensor.size())

    # Get first real tensor
    first_tensor = None
    for tensor in tensors:
        if tensor is not None:
            first_tensor = tensor
            break

    if first_tensor is None:
        raise ValueError
    else:
        assert torch.is_tensor(first_tensor), "This function only works with tensors"
        max_lens = [-1 for _ in range(len(first_tensor.size()))]  # [-1, ..., -1]
        for tensor in tensors:
            if tensor is not None:
                for i in range(len(first_tensor.size())):
                    if max_lens[i] < tensor.size()[i]:
                        max_lens[i] = tensor.size()[i]

        result_tensor_size = [len(tensors)] + max_lens
        create_tensor_fn = first_tensor.new
        result_tensor = create_tensor_fn(*result_tensor_size)
        result_tensor.fill_(value)

        for i, tensor in enumerate(tensors):
            if tensor is not None:
                section = result_tensor[i]
                for dim_idx, dim_size in enumerate(tensor.size()):
                    section = section.narrow(dim_idx, 0, dim_size)
                section.copy_(tensor)
        return result_tensor
