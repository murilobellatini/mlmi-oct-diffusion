import copy
import functools
import os

from datetime import datetime
from time import time
from tqdm import tqdm
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW, lr_scheduler
import wandb
from guided_diffusion.train_sample import sample_images
from guided_diffusion.img_utils import save_images

from guided_diffusion.evaluations.evaluator import compare_sample_images

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        lr_decay,
        lr_stepsize,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        save_only_best=False,
        save_on=None,
        data_valid=None,
        output_interval=None,
        ref_batch_loc=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_train_steps=None,
        max_patience=1000,
        early_stopping_on="loss",
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.data_valid = data_valid
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_stepsize = lr_stepsize
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.output_interval = output_interval
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.max_train_steps = max_train_steps

        self.save_only_best = save_only_best
        self.save_on = save_on
        self.best_metric = float("inf")

        assert (
            not save_only_best or save_on is not None
        ), "Please give a save on metric for best only save"

        self.ref_batch_loc = ref_batch_loc

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self.last_model = None
        self.evaluator = None

        self.max_patience = max_patience
        self.patience = max_patience
        self.early_stopping_on = early_stopping_on
        self.early_stopping_best = float("inf")

        logger.log("Is CUDA available:", self.sync_cuda)

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.lr_scheduler = lr_scheduler.StepLR(
            optimizer=self.opt, step_size=self.lr_stepsize, gamma=self.lr_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
                wandb.log(
                    title="No CUDA Found",
                    text="Distributed training requires CUDA. Gradients will not be synchronized properly!",
                    level=wandb.AlertLevel.WARN,
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self, sampler_fn=None, sample_params=None):
        with tqdm(total=self.max_train_steps - self.resume_step) as pbar:
            while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
            ) and self.patience > 0:
                batch, cond = next(self.data)
                losses = self.run_step(batch, cond)

                if self.data_valid is not None:
                    valid_batch, valid_cond = next(self.data_valid)
                    self.model.eval()
                    valid_losses = self.forward_backward(valid_batch, valid_cond, True)
                    self.model.train()
                    valid_losses = {f"valid_{k}": v for k, v in valid_losses.items()}
                    losses.update(valid_losses)

                if self.early_stopping_on is not None:
                    if (
                        losses[self.early_stopping_on].mean().item()
                        >= self.early_stopping_best
                    ):
                        self.patience -= 1
                    else:
                        self.patience = self.max_patience
                        self.early_stopping_best = (
                            losses[self.early_stopping_on].mean().item()
                        )

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.save_only_best and (
                    self.best_metric > losses[self.save_on].mean().item()
                ):
                    self.best_metric = losses[self.save_on].mean().item()
                    self.save(is_last=False)
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                elif self.step % self.save_interval == 0:
                    self.save(is_last=False)
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                if (
                    self.output_interval is not None
                    and self.output_interval > 0
                    and self.step % self.output_interval == 0
                    and sampler_fn is not None
                    and sample_params is not None
                    and self.last_model is not None
                ):
                    self.save(for_gen=True)
                    sample_params["model_path"] = self.last_model

                    images, _ = sample_images(sample_params)
                    self.model.train()

                    wandb.log(
                        {
                            f"examples_{i}": wandb.Image(image)
                            for i, image in enumerate(images)
                        },
                        step=self.step + self.resume_step,
                    )

                    wandb.log(
                        {
                            "examples": [
                                wandb.Image(image) for i, image in enumerate(images)
                            ]
                        },
                        step=self.step + self.resume_step,
                    )
                    pred_loc = save_images(images, _, False)
                    if self.ref_batch_loc is not None:
                        self.evaluator = compare_sample_images(
                            self.ref_batch_loc, pred_loc, self.evaluator
                        )
                self.step += 1
                pbar.update(1)
                if self.step + self.resume_step >= self.max_train_steps:
                    break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save(is_last=True)
        if self.patience == 0:
            print("Early stopping finished the execution of the training")

    def run_step(self, batch, cond, is_valid=False):
        losses = self.forward_backward(batch, cond, is_valid)
        took_step = self.mp_trainer.optimize(self.opt, self.step + self.resume_step)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        if self.lr_anneal_steps == 0:
            self.lr_scheduler.step()
        self.log_step()
        return losses

    def forward_backward(self, batch, cond, is_valid=False):
        if not is_valid:
            self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler) and not is_valid:
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion,
                t,
                {k: v * weights for k, v in losses.items()},
                self.step + self.resume_step,
                is_valid,
            )
            if not is_valid:
                self.mp_trainer.backward(loss)
        return losses

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step, self.step + self.resume_step)
        logger.logkv(
            "samples",
            (self.step + self.resume_step + 1) * self.global_batch,
            self.step + self.resume_step,
        )

    def save(self, for_gen=False, is_last=False):
        def save_checkpoint(
            rate, params, save_only_best=False, for_gen=False, is_last=False
        ):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    if for_gen:
                        filename = f"model_gen.pt"
                    elif save_only_best and not is_last:  # Still save the last model
                        filename = f"model.pt"
                    else:
                        filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    if for_gen:
                        filename = f"ema_gen.pt"
                    elif save_only_best and not is_last:  # Still save the last model
                        filename = f"ema.pt"
                    else:
                        filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
                    if "model" in filename:
                        self.last_model = bf.join(get_blob_logdir(), filename)

        save_checkpoint(
            0,
            self.mp_trainer.master_params,
            save_only_best=self.save_only_best,
            for_gen=for_gen,
            is_last=is_last,
        )
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(
                rate,
                params,
                save_only_best=self.save_only_best,
                for_gen=for_gen,
                is_last=is_last,
            )

        if dist.get_rank() == 0:
            opt_fname = f"opt{(self.step+self.resume_step):06d}.pt"
            if for_gen:
                opt_fname = "opt_gen.pt"
            elif self.save_only_best and not is_last:
                opt_fname = "opt.pt"
            with bf.BlobFile(
                bf.join(get_blob_logdir(), opt_fname),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses, step=None, is_valid=False):
    for key, values in losses.items():
        logger.logkv_mean("valid_" * is_valid + key, values.mean().item(), step)
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(
                "valid_" * is_valid + f"{key}_q{quartile}", sub_loss, step
            )
