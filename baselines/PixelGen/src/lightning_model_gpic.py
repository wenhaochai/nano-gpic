from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os.path
import copy
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback


import wandb
from torchvision.utils import make_grid
from src.models.autoencoder.base import BaseAE, fp2uint8
from src.models.conditioner.base import BaseConditioner
from src.utils.model_loader import ModelLoader
from src.callbacks.simple_ema import SimpleEMA
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params

torch._functorch.config.donated_buffer = False

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]

SAMPLE_CAPTIONS = [
    "People stand in line at an airport terminal near a large digital flight board showing departures.",
    "cat statue, brick base, tiled ground, garden, outdoor.",
    "People stand on the deck of a large warship under a cloudy sky.",
    "Several vintage typewriters and early computer keyboards are displayed on a wooden desk.",
    "motorcycle, rider, sky, forest, jump",
    "A stone bell tower with four visible bells stands under a clear blue sky.",
    "Five men are posing closely together in a group photo at night, all smiling.",
    "A brown tile with a golden lion design and decorative patterns.",
    "A dog sitting on green grass in a park.",
    "A teacup on a table.",
    "A white boat floating on a red lake.",
    "A sunflower under a blue sky."
]

CFG_SCALES = [1.0, 1.5, 2.0, 4.0, 7.5]

class LightningModel(pl.LightningModule):
    def __init__(self,
                 vae: BaseAE,
                 conditioner: BaseConditioner,
                 denoiser: nn.Module,
                 diffusion_trainer: BaseTrainer,
                 diffusion_sampler: BaseSampler,
                 ema_tracker: SimpleEMA=None,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 eval_original_model: bool = False,
                 sample_every: int = 0,
                 ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.eval_original_model = eval_original_model
        self.sample_every = sample_every

        self._strict_loading = False

    @torch.no_grad()
    def _sample_and_log(self):
        device = next(self.ema_denoiser.parameters()).device
        # all ranks must participate in conditioner forward
        condition, uncondition = self.conditioner(SAMPLE_CAPTIONS)
        
        if self.trainer.global_rank != 0:
            return  # only rank 0 does sampling + logging
        
        B = len(SAMPLE_CAPTIONS)
        latent_shape = self.trainer.datamodule.eval_dataset.latent_shape
        sample_log = {}
        original_guidance = self.diffusion_sampler.guidance
        for cfg in CFG_SCALES:
            self.diffusion_sampler.guidance = cfg
            xT = torch.randn(B, *latent_shape, device=device)
            samples = self.diffusion_sampler(self.ema_denoiser, xT, condition, uncondition)
            samples = self.vae.decode(samples)
            samples = fp2uint8(samples).float()/255.0
            grid = make_grid(samples, nrow=2)
            sample_log[f"samples/cfg_{cfg}"] = wandb.Image(grid.permute(1, 2, 0).cpu().numpy())
        self.diffusion_sampler.guidance = original_guidance
        self.logger.experiment.log(sample_log, step=self.global_step)

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        # disable grad for conditioner and vae
        no_grad(self.conditioner)
        no_grad(self.vae)
        # no_grad(self.diffusion_sampler)
        no_grad(self.ema_denoiser)

        # torch.compile
        print("Compiling models...")
        self.denoiser.compile()
        self.ema_denoiser.compile()
        print("Model compilation done.")

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """perf C3: Lightning's mixed-precision plugin refuses to clip when the optimizer is fused (it assumes the
        optimizer unscales gradients), although bf16 autocast has no gradient scaler. Clip exactly as Lightning's
        'norm' algorithm does: the global 2-norm over every parameter of the optimizer."""
        if not gradient_clip_val:
            return
        assert gradient_clip_algorithm in (None, "norm"), gradient_clip_algorithm
        params = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, gradient_clip_val, foreach=True)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        params_sampler = filter_nograd_tensors(self.diffusion_sampler.parameters())
        param_groups = [
            {"params": params_denoiser, },
            {"params": params_trainer,},
            {"params": params_sampler, "lr": 1e-3},
        ]
        # optimizer: torch.optim.Optimizer = self.optimizer([*params_trainer, *params_denoiser])
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler={
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "learning_rate"
                }
            )

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    # sanity check before training start
    # def on_train_start(self) -> None:
    #     self.ema_denoiser.to(torch.float32)
    #     self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)

    def on_train_start(self) -> None:
        print(f"[rank {self.trainer.global_rank}] on_train_start")
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)
        print(f"[rank {self.trainer.global_rank}] on_train_start done")

    def on_load_checkpoint(self, checkpoint):
        keys_to_check = [
            "denoiser.pos_embed", 
            "ema_denoiser.pos_embed"
        ]
        ckpt_state_dict = checkpoint["state_dict"]
      
        current_state_dict = self.state_dict()

        for key in keys_to_check:
            if key in ckpt_state_dict and key in current_state_dict:
                ckpt_shape = ckpt_state_dict[key].shape
                curr_shape = current_state_dict[key].shape
                if ckpt_shape != curr_shape:
                    print(f"[Warning] Shape mismatch for '{key}': "
                          f"Checkpoint {ckpt_shape} vs Current {curr_shape}. "
                          f"Dropping from checkpoint to avoid RuntimeError.")
                    del ckpt_state_dict[key]
                else:
                    pass

    def training_step(self, batch, batch_idx):
        # print(f"Received training batch with batch_idx={batch_idx}")
        x, y, metadata = batch
        # print(f"Training step: batch_idx={batch_idx}, x.shape={x.shape}, y.shape={y.shape if isinstance(y, torch.Tensor) else 'N/A'},")
        if metadata is None:
            metadata = {}
        metadata['global_step'] = self.global_step
        with torch.no_grad():
            x = self.vae.encode(x)
            condition, uncondition = self.conditioner(y, metadata)
        loss = self.diffusion_trainer(self.denoiser, self.ema_denoiser, self.diffusion_sampler, x, condition, uncondition, metadata)

        # --- GPIC: log per-caption-type training loss ---
        # loss_per_sample is a (B,) tensor returned by the diffusion trainer
        # caption_type is a list of strings e.g. ["short", "tag", "long", ...]
        caption_types = metadata.get("caption_type", None)
        loss_per_sample = loss.pop("loss_per_sample", None)

        # perf C2: the per-caption-type losses need host syncs (index tensors from Python lists) and four sync_dist
        # all-reduces; compute them only on steps whose on_step logs are written (Lightning's own condition), so the
        # logged values are unchanged and every other step runs without them
        log_now = self.logger is not None and self.trainer._logger_connector.should_update_logs
        if log_now and caption_types is not None and loss_per_sample is not None:
            CAPTION_TYPES = ["tag", "short", "medium", "long"]
            for ct in CAPTION_TYPES:
                mask = [i for i, t in enumerate(caption_types) if t == ct]
                if mask:
                    ct_loss = loss_per_sample[torch.tensor(mask, device=loss_per_sample.device)].mean()
                else:
                    ct_loss = torch.tensor(0.0, device=loss_per_sample.device)
                self.log(f"train/loss_{ct}", ct_loss, on_step=True, sync_dist=True)

        # --- end per-caption-type ---

        self.log_dict(loss, prog_bar=False, on_step=True, sync_dist=False)   # perf C2: no per-step .item() for the bar
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, sync_dist=False)
        # if self.trainer.global_rank == 0:
        #     print(f"[step {self.global_step}] loss={loss['loss'].item():.4f}")

        if (self.sample_every > 0
                and self.global_step % self.sample_every == 0
                and self.global_step > 0
                and self.trainer.global_rank == 0):
            self._sample_and_log()

        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y)

        # sample images
        if self.eval_original_model:
            samples = self.diffusion_sampler(self.denoiser, xT, condition, uncondition)
        else:
            samples = self.diffusion_sampler(self.ema_denoiser, xT, condition, uncondition)

        samples = self.vae.decode(samples)
        # fp32 -1,1 -> uint8 0,255
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # --- GPIC: dataloader_idx=0 is GPIC val set (compute loss) ---
        #           dataloader_idx=1 is GenEval (generate images, existing behaviour)
        # print(f"Validation step: batch_idx={batch_idx}, dataloader_idx={dataloader_idx}")
        if dataloader_idx == 0:
            x, y, metadata = batch
            if metadata is None:
                metadata = {}
            with torch.no_grad():
                x = self.vae.encode(x)
                condition, uncondition = self.conditioner(y, metadata)
                loss = self.diffusion_trainer(self.denoiser, self.ema_denoiser, self.diffusion_sampler, x, condition, uncondition, metadata)

            # log overall val loss
            self.log("val/loss", loss["loss"], on_step=False, on_epoch=True, sync_dist=True)

            # log per-caption-type val loss
            caption_types = metadata.get("caption_type", None)
            loss_per_sample = loss.get("loss_per_sample", None)
            if caption_types is not None and loss_per_sample is not None:
                CAPTION_TYPES = ["tag", "short", "medium", "long"]
                for ct in CAPTION_TYPES:
                    mask = [i for i, t in enumerate(caption_types) if t == ct]
                    if mask:
                        ct_loss = loss_per_sample[torch.tensor(mask, device=loss_per_sample.device)].mean()
                        self.log(f"val/loss_{ct}", ct_loss, on_step=False, on_epoch=True, sync_dist=True)
            return loss["loss"]
        # --- end GPIC val loss ---

        # dataloader_idx=1: GenEval image generation (original behaviour)
        samples = self.predict_step(batch, batch_idx)
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix+"denoiser.",
            keep_vars=keep_vars)
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix+"ema_denoiser.",
            keep_vars=keep_vars)
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix+"diffusion_trainer.",
            keep_vars=keep_vars)
        return destination