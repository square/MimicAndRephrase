import torch
import torch.nn as nn
from typing import Sequence, List, Optional, Union, cast, TypeVar, Callable
import math
from log import info


LARGE_NEGATIVE = -1e20



def get_device(var: Union[torch.Tensor, torch.Tensor]) -> int:
    """
    :param var: target variable
    :return: the device a variable is on (>= 0), or -1 if on cpu.
    """
    return var.get_device() if var.is_cuda else -1
