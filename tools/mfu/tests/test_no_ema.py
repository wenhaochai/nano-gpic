#!/usr/bin/env python3
"""GPIC_MFU_NO_EMA=1 (src/callbacks/simple_ema.py): the EMA callback never runs its update; without the switch it runs once
per optimizer step as before. Five simulated optimizer steps through on_train_batch_end (ema_step replaced by a counter, so
no GPU is needed); the EMA parameters keep their values with the switch on. Also on_train_start and state_dict work.
usage: python test_no_ema.py"""
import os, sys, types
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.callbacks.simple_ema import SimpleEMA  # noqa: E402

def run(off):
    if off:
        os.environ["GPIC_MFU_NO_EMA"] = "1"
    else:
        os.environ.pop("GPIC_MFU_NO_EMA", None)
    cb = SimpleEMA(decay=0.9996)
    net, ema = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    cb.setup_models(net, ema)
    before = [p.detach().clone() for p in ema.parameters()]
    calls = []
    cb.ema_step = lambda: calls.append(1)
    tr = types.SimpleNamespace(global_step=0)
    cb.on_train_start(tr, None)
    for s in range(1, 6):
        tr.global_step = s
        cb.on_train_batch_end(tr, None, None, None, s)
        cb.on_train_batch_end(tr, None, None, None, s)   # a second micro-batch of the same step: no second update
    same = all(torch.equal(a, b) for a, b in zip(before, ema.parameters()))
    return len(calls), same, cb.state_dict()

n_off, same_off, sd = run(True)
n_on, _, _ = run(False)
ok = n_off == 0 and same_off and n_on == 5 and sd == {"decay": 0.9996, "every_n_steps": 1}
print(f"switch on: {n_off} EMA updates in 5 steps, EMA unchanged {same_off}; switch off: {n_on} updates; state_dict {sd}")
print("NO_EMA_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
