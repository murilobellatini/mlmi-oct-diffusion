import io
import os

import blobfile as bf
import torch as th
import torch.distributed as dist

GPU_INDEX = 0


def setup_dist():
    # comm = MPI.COMM_WORLD
    # os.environ["RANK"] = str(comm.rank)
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend=backend, init_method="env://")


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
