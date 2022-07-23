import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist


def setup_dist():
    # comm = MPI.COMM_WORLD
    # os.environ["RANK"] = str(comm.rank)
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_PORT"] = str(_find_free_port())
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    if th.cuda.is_available():
        return th.device("cuda")
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


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
