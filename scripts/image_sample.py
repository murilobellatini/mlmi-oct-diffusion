"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
from glob import glob
import os
from pathlib import Path
import click

import numpy as np
import torch as th
import torch.distributed as dist
import wandb
import yaml

from guided_diffusion import seq_utils, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.train_sample import sample_images
from guided_diffusion.img_utils import save_images


@click.command()
@click.argument("params_file", type=click.File("r"))
@click.argument("model_path", type=str)
@click.option(
    "--output_steps",
    is_flag=True,
    show_default=True,
    default=False,
    help="Saves each diff step output",
)
def main(params_file, model_path, output_steps):
    params = get_default_params_sample()

    params_file = yaml.safe_load(params_file)

    params.update(params_file["sample"])
    params.update(params_file["model"])
    params.update(params_file["diffusion"])

    wandb.login(key="f39476c0f8e0beb983d944d595be8f921ec05bfe")
    wandb.init(project="OCT_DM_SAMPLE", entity="mlmioct22", config=params)

    for model_file_path in glob(model_path):
        if not os.path.isfile(model_file_path):
            logger.warn(
                f"No file found in the {model_path} Path. Make sure to pass in a valid .pt file or a valid pattern"
            )
            continue

        params["model_path"] = model_file_path
        model_name = Path(model_file_path).stem

        # dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")

        arr, label_arr = sample_images(params, output_steps=output_steps)

        save_images(arr, label_arr, params["class_cond"], model_name)

        dist.barrier()
        logger.log(f"sampling complete for model {model_name}")


def get_default_params_sample():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    return defaults


if __name__ == "__main__":
    main()
