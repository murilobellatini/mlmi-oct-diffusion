"""
Train a diffusion model on images.
"""

import json
import argparse
import yaml
import click

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

@click.command()
@click.argument("params", type=click.File("r"))
def main(params_file):
    params = get_default_params()
    params.update(yaml.safe_load(params_file))

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    logger.log(f"loaded arguments are: {json.dumps(vars(params), indent=4)}")
    
    model, diffusion = create_model_and_diffusion(
        **params
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        params.schedule_sampler, diffusion
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=params["data_dir"],
        batch_size=params["batch_size"],
        image_size=params["image_size"],
        class_cond=params["class_cond"],
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=params["batch_size"],
        microbatch=params["microbatch"],
        lr=params["lr"],
        ema_rate=params["ema_rate"],
        log_interval=params["log_interval"],
        save_interval=params["save_interval"],
        resume_checkpoint=params["resume_checkpoint"],
        use_fp16=params["use_fp16"],
        fp16_scale_growth=params["fp16_scale_growth"],
        schedule_sampler=schedule_sampler,
        weight_decay=params["weight_decay"],
        lr_anneal_steps=params["lr_anneal_steps"],
        max_train_steps=params["max_train_steps"]
    ).run_loop()


def get_default_params():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        max_train_steps=100
    )
    defaults.update(model_and_diffusion_defaults())
    # parser = argparse.ArgumentParser()
    # add_dict_to_argparser(parser, defaults)
    return defaults


if __name__ == "__main__":
    main()
