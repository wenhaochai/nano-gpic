"""The training objective on a fixed validation set, for comparing optimizers and recipes at matched steps.

Every `every` steps (and after the first step) the flow-matching loss of the raw and of the EMA denoiser is computed
exactly as JiTTrainer computes it (x-prediction, error in v space, its loss weight and t clipping), on fixed data:
the first `num_images` samples of the GPIC validation shards with their own captions (no caption dropout), `num_t`
noise levels at the quantiles of the training t distribution (logit-normal P_mean, P_std, then the time shift), and
one fixed noise per (image, level) from a seeded generator. The mean over levels estimates the expected training loss
under the training t distribution, with no sampling noise between evaluations.

Each rank takes a contiguous slice of the images and the sums are all-reduced. The denoisers run through their eager
forward (module.forward), so the compiled training graphs and their guards stay untouched; captions are encoded once
through the compiled text encoder in chunks of the training batch size, the shape it was compiled for. The EMA side
stream is joined first. Results go to the logger (val/loss_raw, val/loss_ema, val/loss_ema_t{k}) and to
<default_root_dir>/val_loss.jsonl.
"""
import glob
import io
import json
import os
import tarfile

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback

VAL_GLOB = "/scratch/gpfs/GROUP/USER/data/gpic/val/gpic_val_*.tar"


def _read_val(n, pattern=VAL_GLOB):
    """The first n (image bytes, caption) samples of the validation shards, in shard order and tar order. Reading
    stops as soon as n samples are complete, so only the head of the first shard is read."""
    out = []
    for path in sorted(glob.glob(pattern)):
        cur_key, cur = None, {}
        with tarfile.open(path) as tar:
            for m in tar:
                if not m.isfile():
                    continue
                key, ext = m.name.rsplit(".", 1)
                if key != cur_key:
                    if cur_key is not None and "json" in cur and len(cur) > 1:
                        out.append(cur)
                        if len(out) == n:
                            break
                    cur_key, cur = key, {}
                cur[ext] = tar.extractfile(m).read()
            else:
                if cur_key is not None and "json" in cur and len(cur) > 1:
                    out.append(cur)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} validation samples under {pattern}")
    return [(d[[e for e in d if e != "json"][0]], json.loads(d["json"])["caption"]) for d in out[:n]]


class FixedValLoss(Callback):
    def __init__(self, every: int = 1000, num_images: int = 512, num_t: int = 8, batch_size: int = 32, seed: int = 0):
        super().__init__()
        self.every, self.num_images, self.num_t, self.batch_size, self.seed = every, num_images, num_t, batch_size, seed
        self.data = None

    def _setup(self, trainer, pl_module):
        from PIL import Image
        from src.data.dataset.gpic_dataset import val_transform
        from src.diffusion.flow_matching.training_JiT import time_shift_fn
        world, rank = trainer.world_size, trainer.global_rank
        lo, hi = rank * self.num_images // world, (rank + 1) * self.num_images // world
        pairs = _read_val(self.num_images)[lo:hi]
        dev = pl_module.device
        x = torch.stack([val_transform(Image.open(io.BytesIO(b)).convert("RGB")) for b, _ in pairs]).to(dev)
        caps = [c for _, c in pairs]
        with torch.no_grad():
            x = pl_module.vae.encode(x)
            ys = []
            for i in range(0, len(caps), self.batch_size):   # the shape the text encoder was compiled for
                chunk = caps[i:i + self.batch_size]
                pad = self.batch_size - len(chunk)
                cond, _ = pl_module.conditioner(chunk + [""] * pad, {})
                ys.append(cond[:len(chunk)])
            y = torch.cat(ys)
        tr = pl_module.diffusion_trainer
        q = (torch.arange(self.num_t, dtype=torch.float64) + 0.5) / self.num_t
        base_t = torch.sigmoid(tr.P_mean + tr.P_std * torch.special.ndtri(q)).float()
        ts = time_shift_fn(base_t, tr.timeshift).to(dev)
        noise = []
        for j in range(lo, hi):   # one noise per (image, level), independent of world size
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + j)
            noise.append(torch.randn((self.num_t, *x.shape[1:]), generator=g))
        noise = torch.stack(noise, 1).to(dev)   # [num_t, n_local, C, H, W]
        self.data = dict(x=x, y=y, ts=ts, noise=noise, n_local=hi - lo)

    @torch.no_grad()
    def _loss(self, net, tr):
        d = self.data
        sums = torch.zeros(self.num_t, device=d["x"].device, dtype=torch.float64)
        for k in range(self.num_t):
            for i in range(0, d["n_local"], self.batch_size):
                x, y = d["x"][i:i + self.batch_size], d["y"][i:i + self.batch_size]
                t = d["ts"][k].expand(x.shape[0])
                alpha, sigma = tr.scheduler.alpha(t), tr.scheduler.sigma(t)
                x_t = alpha * x + d["noise"][k, i:i + self.batch_size] * sigma
                v_t = (x - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(tr.t_eps)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = net.forward(x_t, t, y)
                out = (pred.float() - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(tr.t_eps)
                loss = tr.loss_weight_fn(alpha, sigma) * (out - v_t) ** 2
                sums[k] += loss.mean(dim=(1, 2, 3)).double().sum()
        return sums

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if not (step == 1 or step % self.every == 0):
            return
        if self.data is None:
            self._setup(trainer, pl_module)
        join = getattr(pl_module.ema_tracker, "_join", None)
        if join is not None:
            join()
        tr = pl_module.diffusion_trainer
        raw = self._loss(pl_module.denoiser, tr)
        ema = self._loss(pl_module.ema_denoiser, tr)
        cnt = torch.tensor([float(self.data["n_local"])], device=raw.device, dtype=torch.float64)
        both = torch.cat([raw, ema, cnt])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(both)
        n = both[-1].item()
        raw_t, ema_t = (both[:self.num_t] / n).tolist(), (both[self.num_t:2 * self.num_t] / n).tolist()
        rec = {"step": step, "loss_raw": sum(raw_t) / self.num_t, "loss_ema": sum(ema_t) / self.num_t,
               "loss_raw_t": raw_t, "loss_ema_t": ema_t, "t": self.data["ts"].tolist(), "n": int(n)}
        if trainer.global_rank == 0:
            metrics = {"val/loss_raw": rec["loss_raw"], "val/loss_ema": rec["loss_ema"]}
            metrics.update({f"val/loss_ema_t{k}": v for k, v in enumerate(ema_t)})
            for lg in trainer.loggers:
                lg.log_metrics(metrics, step=step)
            with open(os.path.join(trainer.default_root_dir, "val_loss.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[val] step {step} loss_raw {rec['loss_raw']:.5f} loss_ema {rec['loss_ema']:.5f}", flush=True)
