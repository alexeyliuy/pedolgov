from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def derive_seed(seed: int, salt: str) -> int:
    acc = seed & 0xFFFFFFFF
    for ch in salt:
        acc = (acc * 1000003 + ord(ch)) & 0xFFFFFFFF
    return acc
