"""Test, without a GPU, of launch lever G1 (GPIC_LAUNCH_SKIP_EMA_COPY=1 with GPIC_MFU_NO_EMA=1: configure_model of
baselines/PixelGen/src/lightning_model_gpic.py leaves out the second copy of the weights into the EMA slot) on the real
LightningModel with a small denoiser:
  1  the copy is left out only with both switches; with either one alone, or none, configure_model copies as before
  2  left out, the EMA slot holds what __init__'s deepcopy gave it: the denoiser's weights, bit for bit
  3  even if the denoiser changed after __init__ (nothing does), no reader sees the EMA slot's old values under
     GPIC_MFU_NO_EMA: state_dict (every checkpoint) first writes the raw weights over the slot (SimpleEMA.mirror)
usage (a vis node, from baselines/PixelGen): PYTHONPATH=. python ../../tools/mfu/tests/test_ema_copy.py
Result: TEST_EMA_COPY PASS: N checks."""
import os
import sys
import types

import torch
import torch.nn as nn

PIXGEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "baselines", "PixelGen")
sys.path.insert(0, os.path.abspath(PIXGEN))
import src.lightning_model_gpic as M   # noqa: E402
from src.callbacks.simple_ema import SimpleEMA   # noqa: E402

FAILS, N = [], [0]
CALLS = []
_real_copy = M.copy_params


def counted(src_model, dst_model):
    CALLS.append(1)
    return _real_copy(src_model=src_model, dst_model=dst_model)


M.copy_params = counted


def check(ok, what):
    N[0] += 1
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def build(env, drift):
    """The model as LightningCLI builds it, then (drift) a change of the denoiser that nothing in a launch makes."""
    for k in ("GPIC_LAUNCH_SKIP_EMA_COPY", "GPIC_MFU_NO_EMA", "GPIC_MFU_FSDP", "GPIC_MFU_FP8"):
        os.environ.pop(k, None)
    os.environ.update(env)
    torch.manual_seed(0)
    den = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    m = M.LightningModel(vae=nn.Identity(), conditioner=nn.Identity(), denoiser=den, diffusion_trainer=nn.Identity(),
                         diffusion_sampler=nn.Identity(), ema_tracker=SimpleEMA(decay=0.99))   # the tracker reads GPIC_MFU_NO_EMA here
    m._trainer = types.SimpleNamespace(strategy=types.SimpleNamespace(barrier=lambda: None))
    init = [p.detach().clone() for p in m.denoiser.parameters()]
    if drift:
        with torch.no_grad():
            for p in m.denoiser.parameters():
                p.add_(1.0)
    del CALLS[:]
    m.configure_model()
    return m, init


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


both = {"GPIC_LAUNCH_SKIP_EMA_COPY": "1", "GPIC_MFU_NO_EMA": "1"}
for name, env, want in (("no switch", {}, 1), ("only GPIC_LAUNCH_SKIP_EMA_COPY=1", {"GPIC_LAUNCH_SKIP_EMA_COPY": "1"}, 1),
                        ("only GPIC_MFU_NO_EMA=1", {"GPIC_MFU_NO_EMA": "1"}, 1), ("GPIC_LAUNCH_SKIP_EMA_COPY=0 with no EMA", {**both, "GPIC_LAUNCH_SKIP_EMA_COPY": "0"}, 1),
                        ("both switches", both, 0)):
    m, init = build(env, drift=True)
    den, ema = list(m.denoiser.parameters()), list(m.ema_denoiser.parameters())
    check(len(CALLS) == want, f"1: {name}: {len(CALLS)} copies (expected {want})")
    if want:
        check(same(den, ema) and not same(ema, init), f"1: {name}: the EMA slot holds the denoiser's weights of now")
    else:
        check(same(ema, init) and not same(den, ema), f"1: {name}: the EMA slot holds the deepcopy's weights, not the changed denoiser's")
    check(all(not p.requires_grad for p in ema), f"1: {name}: the EMA slot has no gradients")

m, init = build(both, drift=False)   # a real launch: nothing changes the denoiser between __init__ and configure_model
den, ema = list(m.denoiser.parameters()), list(m.ema_denoiser.parameters())
check(len(CALLS) == 0 and same(ema, den) and same(ema, init) and all(e is not d for e, d in zip(ema, den)),
      "2: both switches: no copy, and the EMA slot equals the denoiser bit for bit (two sets of tensors)")
m2, _ = build({}, drift=False)
check(len(CALLS) == 1 and same(list(m2.ema_denoiser.parameters()), ema) and same(list(m2.denoiser.parameters()), den),
      "2: the launch with the copy ends configure_model in the same state")

m, init = build(both, drift=True)   # the worst case, which no launch has
check(m.ema_tracker.off, "3: the tracker is off under GPIC_MFU_NO_EMA")
m.ema_tracker.setup_models(net=m.denoiser, ema_net=m.ema_denoiser)   # on_train_start
sd = m.state_dict()
check(all(torch.equal(sd["denoiser." + k], sd["ema_denoiser." + k]) for k in m.denoiser.state_dict()) and same(list(m.ema_denoiser.parameters()), list(m.denoiser.parameters())),
      "3: state_dict writes the raw weights over the EMA slot: the checkpoint's two sets are equal")
print(f"TEST_EMA_COPY {'PASS' if not FAILS else 'FAIL'}: {N[0]} checks" + (f", {len(FAILS)} failed" if FAILS else ""), flush=True)
sys.exit(1 if FAILS else 0)
