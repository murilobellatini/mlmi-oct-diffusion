"""
Train a diffusion model on images.
"""

import json
import wandb
import yaml
import click

from guided_diffusion import dist_util
from guided_diffusion import logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.train_util import TrainLoop
from scripts.image_sample import get_default_params_sample, sample_images


@click.command()
@click.argument("params_file", type=click.File("r"))
@click.option(
    "--drop_data_val",
    is_flag=True,
    show_default=True,
    default=False,
    help="Drops data validation",
)
@click.option(
    "--only_square",
    is_flag=True,
    show_default=True,
    default=False,
    help="Drops data validation",
)
def main(params_file, drop_data_val, only_square):
    params_file = yaml.safe_load(params_file)

    params = get_default_params()
    params.update(params_file["train"])
    params.update(params_file["model"])
    params.update(params_file["diffusion"])
    params.update(params_file["data"])

    sample_params = get_default_params_sample()
    sample_params.update(params_file["sample"])
    sample_params.update(params_file["model"])
    sample_params.update(params_file["diffusion"])
    sample_params.update(params_file["data"])

    wandb.login(key="f39476c0f8e0beb983d944d595be8f921ec05bfe")
    wandb.init(project="OCT_DM_TRAIN", entity="mlmioct22", config=params)

    dist_util.setup_dist()
    logger.configure(dir=params_file["train"].get("output_dir", None), web_logger=True)

    logger.log("creating model and diffusion...")
    logger.log(f"loaded arguments are: {json.dumps(params, indent=4)}")

    model, diffusion = create_model_and_diffusion(
        **{k: params[k] for k in model_and_diffusion_defaults().keys() if k in params}
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        params["schedule_sampler"], diffusion
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=params["data_dir"],
        batch_size=params["batch_size"],
        image_size=params["image_size"],
        class_cond=params["class_cond"],
        resize_image=params["resize_image"],
        resize_res=params["resize_res"],
        random_crop=params["random_crop"],
        center_crop=params["center_crop"],
        validate_data=not drop_data_val,
        only_square=only_square,
    )
    if (
        params.get("valid_data_dir", None) is not None
        and params["valid_data_dir"] != ""
    ):
        data_valid = load_data(
            data_dir=params["valid_data_dir"],
            batch_size=params["batch_size"],
            image_size=params["image_size"],
            class_cond=params["class_cond"],
            resize_image=params["resize_image"],
            resize_res=params["resize_res"],
            random_crop=params["random_crop"],
            center_crop=params["center_crop"],
            validate_data=not drop_data_val,
            only_square=only_square,
        )
    else:
        data_valid = None

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        data_valid=data_valid,
        batch_size=params["batch_size"],
        microbatch=params["microbatch"],
        lr=params["lr"],
        lr_decay=params["lr_decay"],
        lr_stepsize=params["lr_stepsize"],
        ema_rate=params["ema_rate"],
        log_interval=params["log_interval"],
        save_interval=params["save_interval"],
        output_interval=params["output_interval"],
        resume_checkpoint=params["resume_checkpoint"],
        use_fp16=params["use_fp16"],
        fp16_scale_growth=params["fp16_scale_growth"],
        schedule_sampler=schedule_sampler,
        weight_decay=params["weight_decay"],
        lr_anneal_steps=params["lr_anneal_steps"],
        max_train_steps=params["max_train_steps"],
        ref_batch_loc=params.get("reference_samples_path", None),
        save_only_best=params.get("save_only_best_model", False),
        save_on=params.get("save_metric", None),
        max_patience=params.get("max_patience", 1000),
        early_stopping_on=params.get("early_stopping_on", None),
    ).run_loop(sample_images, sample_params)


def get_default_params():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        lr_decay=1.0,  # Default value for learning rate decay set to 1 to keep LR stable by default
        lr_stepsize=10000,  # StepLR Scheduler with defined stepsize for decreasing learning rate
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        output_interval=None,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        max_train_steps=100,
    )
    defaults.update(model_and_diffusion_defaults())
    # parser = argparse.ArgumentParser()
    # add_dict_to_argparser(parser, defaults)
    return defaults


if __name__ == "__main__":
    main()
