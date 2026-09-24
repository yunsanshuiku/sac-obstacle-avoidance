"""Keep training metadata compatible with PyTorch's tensor-only loader."""
import numpy as np


def basic_metadata(value):
    """Replace NumPy metadata recursively, preserving tensors and integer keys.

    Rewards/distances often become NumPy scalars. Converting before saving avoids
    custom pickle globals in otherwise plain SAC checkpoints.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: basic_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [basic_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(basic_metadata(item) for item in value)
    return value
