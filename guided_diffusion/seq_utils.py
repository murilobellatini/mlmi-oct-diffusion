import io

import blobfile as bf
import torch as th

GPU_INDEX = 0


def dev():
    if th.cuda.is_available():
        return th.device(f"cuda:{str(GPU_INDEX)}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    pass
