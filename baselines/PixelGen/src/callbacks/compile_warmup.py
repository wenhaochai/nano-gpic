"""ladder: compile every training graph on all ranks at the same time, before the first training step.

With grouped text the padded length T changes between batches. A new T recompiles the text encoder and the denoiser on
the rank that sees it, and the other ranks wait for that rank at the next all-reduce, so eight ranks compile one after
the other (w916, 2026-11-02: fewer than 80 steps in 14 minutes). This callback runs the real training path (text
encoder, CFG dropout, denoiser forward and backward, under the training precision context) on dummy batches at T = 64,
128, 320 and 192 on every rank together: the second T gives the graphs a dynamic T, 320 is the slice of the whole
position embedding (its own guard), and 192 checks that no new graph is needed. The gradients are discarded before the first step, and DDP's reducer ignores this backward because it runs
before any DDP forward. The only trace left in training is the position of the RNG streams.
"""
import contextlib
import os
import time

import torch
from lightning.pytorch.callbacks import Callback


class CompileWarmup(Callback):
    def __init__(self, batch_size: int = 32, lengths=(64, 128, 320, 192), image_size: int = 256):
        super().__init__()
        self.batch_size, self.lengths, self.image_size = batch_size, tuple(lengths), image_size

    def on_train_start(self, trainer, pl_module):
        # the data's per-GPU batch when the DataModule has one (new ladders: micro-batches of 64 or 128 for global 1024)
        B, dev = getattr(getattr(trainer, "datamodule", None), "train_batch_size", None) or self.batch_size, pl_module.device
        cached = not hasattr(pl_module.conditioner, "tokenizer")   # outputs read from the batch (cached_text.py)
        for T in self.lengths:
            t0 = time.time()
            mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
            mask[:, :T // 2] = True   # with padding, as real batches have: HF skips the mask when nothing is padded
            if cached:
                text = {"text_emb": torch.zeros(B, T, pl_module.conditioner.embed_dim, dtype=torch.bfloat16, device=dev)}
            else:
                pad = pl_module.conditioner.tokenizer.pad_token_id
                text = {"input_ids": torch.full((B, T), pad, dtype=torch.long, device=dev)}
            metadata = {**text, "text_mask": mask, "global_step": trainer.global_step}
            if os.environ.get("GPIC_MFU_PACK") == "1":   # MFU lever E: packed captions, lengths below T as real batches
                from src.utils.rebalance import Rebalancer
                lens = [max(1, T // 2 - (i % 5)) for i in range(B)]
                metadata = {"text_emb": torch.zeros(sum(lens), pl_module.conditioner.embed_dim, dtype=torch.bfloat16,
                                                    device=dev), "text_lens": lens, "global_step": trainer.global_step}
                pack_plan = Rebalancer(rebalance=False).host_plan(lens)
            x = torch.zeros(B, 3, self.image_size, self.image_size, device=dev)
            with trainer.precision_plugin.train_step_context():
                with torch.no_grad():
                    x = pl_module.vae.encode(x)
                    condition, uncondition = pl_module.conditioner([""] * B, metadata)
                    if "text_lens" in metadata:
                        x, condition, metadata["y_pack"] = Rebalancer(rebalance=False).exchange(x, condition, pack_plan)
                # inside DDP's forward context: dynamo checks it at compile time, and only then splits the graph at
                # the DDP buckets (DDPOptimizer), so these are the graphs a real step would compile (review S2)
                ddp_ctx = getattr(trainer.strategy.model, "_inside_ddp_forward", None)   # FSDP2 (MFU lever B): no DDP
                with (ddp_ctx() if ddp_ctx is not None else contextlib.nullcontext()):
                    loss = pl_module.diffusion_trainer(pl_module.denoiser, pl_module.ema_denoiser,
                                                       pl_module.diffusion_sampler, x, condition, uncondition, metadata)
            loss["loss"].backward()
            pl_module.denoiser.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            trainer.strategy.barrier()
            if trainer.is_global_zero:
                print(f"[warmup] T={T}: {time.time() - t0:.1f} s", flush=True)
        if trainer.is_global_zero:
            print("[warmup] done", flush=True)
