from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
import torch


def nest_dict(d: Dict, separator: str = ".") -> Dict:
    nested_d = dict()
    for key in d.keys():
        key_parts = key.split(separator)
        current_d = nested_d
        while len(key_parts) > 1:
            if key_parts[0] not in current_d:
                current_d[key_parts[0]] = dict()
            current_d = current_d[key_parts[0]]
            key_parts.pop(0)
        current_d[key_parts[0]] = d[key]  # Set the value
    return nested_d

def get_from_batch(batch: Any, start: Union[int, np.ndarray, torch.Tensor], end: Optional[int] = None) -> Any:
    if isinstance(batch, (dict, h5py.Group)):
        return {k: get_from_batch(v, start, end=end) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [get_from_batch(v, start, end=end) for v in batch]
    elif isinstance(batch, (np.ndarray, torch.Tensor, h5py.Dataset)):
        if end is None:
            return batch[start]
        else:
            return batch[start:end]
    else:
        raise ValueError("Unsupported type passed to `get_from_batch`")

def remove_float64(batch: Any):
    if isinstance(batch, dict):
        return {k: remove_float64(v) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [remove_float64(v) for v in batch]
    elif isinstance(batch, np.ndarray):
        if batch.dtype == np.float64:
            return batch.astype(np.float32)
    elif isinstance(batch, torch.Tensor):
        if batch.dtype == torch.double:
            return batch.float()
    else:
        raise ValueError("Unsupported type passed to `remove_float64`")
    return batch
