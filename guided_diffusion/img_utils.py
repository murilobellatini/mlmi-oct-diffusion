import os
import numpy as np
import torch.distributed as dist
from guided_diffusion import logger


def save_images(arr, label_arr, class_cond, model_name=""):
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(
            logger.get_dir(),
            f"samples_{shape_str + '_' * (len(model_name) > 0) + model_name}.npz",
        )
        logger.log(f"saving to {out_path}")
        if class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        return out_path
