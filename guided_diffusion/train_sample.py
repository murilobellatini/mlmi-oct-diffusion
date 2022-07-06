import os
from guided_diffusion import dist_util, logger
import torch as th
import numpy as np
import torch.distributed as dist
from guided_diffusion.script_util import (
    NUM_CLASSES,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def sample_images(params, model=None, diffusion=None, seq=False, micro=None, t=None):
    if model is None or diffusion is None:
        model, diffusion = create_model_and_diffusion(
            **{
                k: params[k]
                for k in model_and_diffusion_defaults().keys()
                if k in params
            }
        )
        model.load_state_dict(
            dist_util.load_state_dict(params["model_path"], seq=seq, map_location="cpu")
        )
        model.to(dist_util.dev())

        if params["use_fp16"]:
            model.convert_to_fp16()
    model.eval()
    print(1, model(micro, t))
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
                device=dist_util.dev(),
            )
            model_kwargs["y"] = classes
            print(2, model(micro, t))
        sample_fn = (
            diffusion.p_sample_loop
            if not params["use_ddim"]
            else diffusion.ddim_sample_loop
        )
        print(3, model(micro, t))
        sample = sample_fn(
            model,
            (params["batch_size"], 3, params["image_size"], params["image_size"]),
            clip_denoised=params["clip_denoised"],
            model_kwargs=model_kwargs,
        )
        print(4, model(micro, t))
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        print(5, model(micro, t))
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        print(6, model(micro, t))
        if params["class_cond"]:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * params['batch_size']} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: params["num_samples"]]
    print(7, model(micro, t))

    if params["class_cond"]:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: params["num_samples"]]
        return arr, label_arr
    else:
        return arr, None


def save_images(arr, label_arr, class_cond):
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        return out_path
