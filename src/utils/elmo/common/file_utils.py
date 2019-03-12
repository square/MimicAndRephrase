"""
Utilities for working with the local dataset cache.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union, IO, Callable


def cached_path(url_or_filename: Union[str, Path], cache_dir: str = None) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if os.path.exists(url_or_filename):
        return url_or_filename
    else:
        raise FileNotFoundError("file {} not found".format(url_or_filename))


def get_file_extension(path: str, dot=True, lower: bool = True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext
