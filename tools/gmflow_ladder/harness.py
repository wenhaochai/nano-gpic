"""CPU harness of the final ladder's training path for tools/gmflow_ladder/test_port.py: the real LightningModel with a tiny JiT_T2I
(point, or the per-pixel mixture with gm_k), the trainer of the run (JiTTrainer or GMFlowTrainer, lambda 0.5), MuonAdamW with the ladder's
settings (configs_h100/muon_ladder.yaml; Muon on the block matrices, AdamW on the rest), GPIC_MFU_NO_EMA=1 (SimpleEMA off, the EMA slot mirrors
the raw weights), WarmupFlatCooldown with warmup 1%, a linear cooldown over the last 50% to min_lr 3e-5, gradient clip 1.0 (norm), bf16-mixed,
the launcher's two CheckpointHooks (ms and last) and CompileWarmup (with its resume reseed), and a CSV logger that writes every step (so the
mixture's diagnostics and the pre-clip norm are computed). torch.compile stays as configure_model calls it unless TORCHDYNAMO_DISABLE=1.

It imports only src.* (PYTHONPATH picks the code tree), and builds the point model without any mixture argument, so the same file runs on
5ac69ab (test (d)). usage: python harness.py <json config> ; the config: out (a .pt it writes), root (run directory), k (0 = point model), steps
(max_steps), resume (a checkpoint or null), seed, batch, ms_every, last_every, probe (true: the loss of a fixed batch at the end of training
and at the start of a resumed training, under its own seeds).
Writes: losses (per training step: the trainer's loss), lrs (per step: each param group's lr), probe_end / probe_start, groups (per param
group: its hyperparameters and the names of its parameters), state (the denoiser's state_dict at the end), ema_equal (EMA slot == raw in
every checkpoint written), saves (the files written)."""
import glob
import json
import os
import sys

import torch

cfg = json.load(open(sys.argv[1]))
os.environ["GPIC_MFU_NO_EMA"] = "1"
for k in ("GPIC_MFU_FSDP", "GPIC_MFU_FUSED_OPT", "GPIC_MFU_ATTN", "GPIC_MFU_FUSED_NORM", "GPIC_MFU_FUSED_QK", "GPIC_MFU_MUON_ADD"):
    os.environ.pop(k, None)   # GPU-only levers: SDPA, the unfused norm, torch's add (the same math up to rounding, tested elsewhere)
os.environ["PL_GLOBAL_SEED"] = str(cfg.get("seed", 0))
if not torch.cuda.is_available():   # CompileWarmup and SimpleEMA synchronize the GPU's streams; on a CPU there is nothing to wait for
    import types
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.current_stream = lambda *a, **k: types.SimpleNamespace(synchronize=lambda: None, wait_stream=lambda s: None)

import lightning.pytorch as pl   # noqa: E402
from functools import partial   # noqa: E402
from lightning.pytorch.loggers import CSVLogger   # noqa: E402

from src.callbacks.compile_warmup import CompileWarmup   # noqa: E402
from src.callbacks.model_checkpoint import CheckpointHook   # noqa: E402
from src.callbacks.simple_ema import SimpleEMA   # noqa: E402
from src.diffusion.base.guidance import simple_guidance_fn   # noqa: E402
from src.diffusion.flow_matching.sampling import EulerSamplerJiT   # noqa: E402
from src.diffusion.flow_matching.scheduling import LinearScheduler   # noqa: E402
from src.diffusion.flow_matching.training_JiT import JiTTrainer, constant   # noqa: E402
from src.lightning_model_gpic import LightningModel   # noqa: E402
from src.models.autoencoder.pixel import PixelAE   # noqa: E402
from src.models.conditioner.base import BaseConditioner   # noqa: E402
from src.models.transformer.JiT_T2I import JiT_T2I   # noqa: E402
from src.optim.muon import MuonAdamW   # noqa: E402
from src.utils.lr_scheduler import WarmupFlatCooldown   # noqa: E402

E, IMG, TMAX = 32, 64, 32
SMALL = dict(input_size=IMG, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=E,
             txt_max_length=TMAX, bottleneck_dim=16)
K, STEPS, B = int(cfg["k"]), int(cfg["steps"]), int(cfg.get("batch", 4))
TOTAL = int(cfg.get("total_steps", STEPS))   # the schedule's max_steps (a run cut into launches keeps it)


class TinyCond(BaseConditioner):
    """CachedTextConditioner on the CPU: the batch's cached outputs, the mask into metadata["y_mask"], a zero null caption."""
    def __init__(self):
        super().__init__()
        self.embed_dim = E

    def _impl_condition(self, y, metadata):
        metadata["y_mask"] = metadata["text_mask"].bool()
        return metadata["text_emb"]

    def _impl_uncondition(self, y, metadata):
        return torch.zeros_like(metadata["text_emb"])


class Data(torch.utils.data.Dataset):
    """Sample i: a fixed image in [-1, 1] and a caption of 5 + i % 11 tokens, from its own seed."""
    def __len__(self):
        return 10 ** 6

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(1000 + i)
        n = 5 + i % 11
        return torch.rand(3, IMG, IMG, generator=g) * 2 - 1, n, torch.randn(n, E, generator=g)


def collate(items):
    T = -(-max(n for _, n, _ in items) // 8) * 8
    emb = torch.zeros(len(items), T, E, dtype=torch.bfloat16)
    mask = torch.zeros(len(items), T, dtype=torch.bool)
    for r, (_, n, e) in enumerate(items):
        emb[r, :n], mask[r, :n] = e.bfloat16(), True
    return torch.stack([x for x, _, _ in items]), [""] * len(items), {"text_emb": emb, "text_mask": mask, "caption_type": ["short"] * len(items)}


def build():
    pl.seed_everything(int(cfg.get("seed", 0)), workers=True)
    den = JiT_T2I(**SMALL, **({"gm_k": K} if K else {}))
    targs = dict(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, P_mean=-0.8, P_std=0.8, t_eps=0.05,
                 null_condition_p=0.0)
    if K:
        from src.diffusion.flow_matching.training_gmflow import GMFlowTrainer
        tr = GMFlowTrainer(**targs, lam=0.5, weight_scale=2.0)
    else:
        tr = JiTTrainer(**targs)
    smp = EulerSamplerJiT(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=4, guidance=1.0)
    mlr, alr = float(cfg.get("muon_lr", 1.36e-2)), float(cfg.get("adamw_lr", 5.01e-4))
    opt = partial(MuonAdamW, lr=alr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, muon_lr=mlr, muon_weight_decay=5 / (mlr * TOTAL),
                  muon_momentum=0.95, muon_nesterov=True, ns_steps=5, split=True, update_scale="mup", fused_adamw=True, ns_dtype="bfloat16")
    sch = WarmupFlatCooldown(num_warmup_steps=max(1, TOTAL // 100), num_cooldown_steps=(TOTAL * 50 // 100) // 32 * 32 or TOTAL // 2,
                  max_steps=TOTAL, min_lr=3e-5)
    return LightningModel(vae=PixelAE(scale=1.0), conditioner=TinyCond(), denoiser=den, diffusion_trainer=tr, diffusion_sampler=smp,
                          ema_tracker=SimpleEMA(decay=0.9996), optimizer=opt, lr_scheduler=sch)


PROBE = {}


def probe(m):
    """The trainer's loss on a fixed batch, under its own seeds, without a gradient and without moving any random stream."""
    x, y, meta = collate([Data()[10 ** 5 + i] for i in range(B)])
    with torch.random.fork_rng(devices=[]), torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.manual_seed(12345)
        cond, unc = m.conditioner(y, meta)
        return float(m.diffusion_trainer(m.denoiser, m.ema_denoiser, m.diffusion_sampler, m.vae.encode(x), cond, unc, meta)["loss"])


class Record(pl.Callback):
    def __init__(self):
        self.losses, self.lrs = [], []

    def on_train_start(self, trainer, m):
        if cfg.get("probe"):
            PROBE["start"] = (trainer.global_step, probe(m))

    def on_train_batch_end(self, trainer, m, outputs, batch, i):
        self.losses.append(float(outputs["loss"]))
        self.lrs.append([g["lr"] for g in trainer.optimizers[0].param_groups])

    def on_train_end(self, trainer, m):
        if cfg.get("probe"):
            PROBE["end"] = (trainer.global_step, probe(m))


def main():
    m = build()
    root = cfg["root"]
    rec = Record()
    cbs = [CheckpointHook(dirpath=root, filename="ms-{step:06d}", monitor=None, save_last=False, save_top_k=-1, every_n_train_steps=int(cfg["ms_every"])),
           CheckpointHook(dirpath=root, filename="resume-{step:06d}", monitor=None, save_last=True, save_top_k=0, every_n_train_steps=int(cfg["last_every"]),
                          enable_version_counter=False),
           CompileWarmup(batch_size=B, lengths=(8, 16), image_size=IMG), rec]
    trainer = pl.Trainer(default_root_dir=root, accelerator="cpu", devices=1, precision="bf16-mixed", max_steps=STEPS,   # CheckpointHook.setup writes into default_root_dir
                         gradient_clip_val=1.0, gradient_clip_algorithm="norm",
                         accumulate_grad_batches=1, logger=CSVLogger(root, name="csv"), log_every_n_steps=1, callbacks=cbs,
                         enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0)
    dl = torch.utils.data.DataLoader(Data(), batch_size=B, shuffle=False, collate_fn=collate, num_workers=0)
    trainer.fit(m, train_dataloaders=dl, ckpt_path=cfg.get("resume"))
    names = {id(p): n for n, p in m.named_parameters()}
    groups = [{"hyper": {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items() if k != "params" and isinstance(v, (int, float, bool, str, tuple, type(None)))},
               "names": [names[id(p)] for p in g["params"]]} for g in trainer.optimizers[0].param_groups]
    saves = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "*.ckpt")))
    ema_equal = {}
    for p in glob.glob(os.path.join(root, "*.ckpt")):
        sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        ema_equal[os.path.basename(p)] = all(torch.equal(v, sd["ema_denoiser." + k[len("denoiser."):]]) for k, v in sd.items() if k.startswith("denoiser."))
    torch.save(dict(losses=rec.losses, lrs=rec.lrs, probe_end=PROBE.get("end"), probe_start=PROBE.get("start"), groups=groups, saves=saves,
                    ema_equal=ema_equal, ema_off=bool(m.ema_tracker.off), global_step=trainer.global_step,
                    state={k: v.detach().clone() for k, v in m.denoiser.state_dict().items()}), cfg["out"])
    print(f"[harness] k={K} steps {trainer.global_step} losses {['%.5f' % v for v in rec.losses]}", flush=True)


if __name__ == "__main__":
    main()
