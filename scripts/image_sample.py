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


def sample_images(params):
    model, diffusion = create_model_and_diffusion(
        **{k: params[k] for k in model_and_diffusion_defaults().keys() if k in params}
    )
    model.load_state_dict(
        dist_util.load_state_dict(params["model_path"], map_location="cpu")
    )
    model.to(dist_util.dev())
    if params["use_fp16"]:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * params["batch_size"] < params["num_samples"]:
        model_kwargs = {}
        if params["class_cond"]:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(params["batch_size"],), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not params["use_ddim"] else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (params["batch_size"], 3, params["image_size"], params["image_size"]),
            clip_denoised=params["clip_denoised"],
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if params["class_cond"]:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * params['batch_size']} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: params["num_samples"]]

    if params["class_cond"]:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: params["num_samples"]]
        return arr, label_arr
    else:
        return arr, None



@click.command()
@click.argument("params_file", type=click.File("r"))
def main(params_file):
    params = get_default_params_sample()

    params_file = yaml.safe_load(params_file)
    
    params.update(params["sample"])
    params.update(params["model"])
    params.update(params["diffusion"])

    wandb.login(key="f39476c0f8e0beb983d944d595be8f921ec05bfe")
    wandb.init(project="OCT DM", entity="mlmioct22")
    wandb.config = params

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    arr, label_arr = sample_images()
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if params["class_cond"]:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

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
