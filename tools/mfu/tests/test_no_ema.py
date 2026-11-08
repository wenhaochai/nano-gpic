#!/usr/bin/env python3
"""GPIC_MFU_NO_EMA=1 (src/callbacks/simple_ema.py; owner 2026-11-08: 「ema是一定不能用」): the EMA callback never updates,
and mirror() (called by the module's state_dict before every checkpoint) makes the EMA parameters equal to the raw ones, so
every checkpoint's EMA slot holds the raw weights. Without the switch: one update per optimizer step, mirror() a no-op.
Five simulated optimizer steps (ema_step counted, no GPU needed). usage: python test_no_ema.py"""
import os, sys, types
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.callbacks.simple_ema import SimpleEMA  # noqa: E402


def run(off):
    if off:
        os.environ["GPIC_MFU_NO_EMA"] = "1"
    else:
        os.environ.pop("GPIC_MFU_NO_EMA", None)
    torch.manual_seed(0)
    cb = SimpleEMA(decay=0.9996)
    net, ema = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    cb.setup_models(net, ema)
    calls = []
    cb.ema_step = lambda: calls.append(1)
    tr = types.SimpleNamespace(global_step=0)
    cb.on_train_start(tr, None)
    for s in range(1, 6):
        tr.global_step = s
        cb.on_train_batch_end(tr, None, None, None, s)
    before = [p.detach().clone() for p in ema.parameters()]
    cb.mirror()
    mirrored = all(torch.equal(a, b) for a, b in zip(ema.parameters(), net.parameters()))
    untouched = all(torch.equal(a, b) for a, b in zip(ema.parameters(), before))
    return len(calls), mirrored, untouched


n_off, mir_off, _ = run(True)
n_on, mir_on, untouched_on = run(False)
ok = n_off == 0 and mir_off and n_on == 5 and untouched_on and not mir_on
print(f"switch on: {n_off} EMA updates, EMA == raw after mirror {mir_off}; switch off: {n_on} updates, mirror left the EMA "
      f"as it was {untouched_on}")
print("NO_EMA_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
