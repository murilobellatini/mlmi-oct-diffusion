"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import click

import numpy as np
import torch as th
import torch.distributed as dist
import wandb
import yaml

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.train_sample import sample_images, save_images


@click.command()
@click.argument("params_file", type=click.File("r"))
def main(params_file):
    params = get_default_params_sample()

    params_file = yaml.safe_load(params_file)

    params.update(params["sample"])
    params.update(params["model"])
    params.update(params["diffusion"])

    wandb.login(key="f39476c0f8e0beb983d944d595be8f921ec05bfe")
    wandb.init(project="OCT_DM_SAMPLE", entity="mlmioct22")
    wandb.config = params

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    arr, label_arr = sample_images()

    save_images(arr, label_arr, params["class_cond"])

    dist.barrier()
    logger.log("sampling complete")


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