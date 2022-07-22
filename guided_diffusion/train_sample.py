import os
from guided_diffusion import seq_utils, logger
import torch as th
import numpy as np
import torch.distributed as dist
from guided_diffusion.script_util import (
    NUM_CLASSES,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def sample_images(params, model=None, diffusion=None, output_steps=False):
    if model is None or diffusion is None:
        model, diffusion = create_model_and_diffusion(
            **{
                k: params[k]
                for k in model_and_diffusion_defaults().keys()
                if k in params
            }
        )
        model.load_state_dict(
            seq_utils.load_state_dict(params["model_path"], map_location="cpu")
        )
        model.to(seq_utils.dev())

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
                low=0,
                high=NUM_CLASSES,
                size=(params["batch_size"],),
                device=seq_utils.dev(),
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop
            if not params["use_ddim"]
            else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (params["batch_size"], 3, params["image_size"], params["image_size"]),
            clip_denoised=params["clip_denoised"],
            model_kwargs=model_kwargs,
            output_steps=output_steps,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([s.cpu().numpy() for s in gathered_samples])

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
