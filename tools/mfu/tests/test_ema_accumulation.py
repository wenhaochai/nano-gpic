#!/usr/bin/env python3
"""src/callbacks/simple_ema.py under gradient accumulation (the Muon ladder: global batch 1024 = 8 x 32 x 4): the EMA
moves once per optimizer step, also right after a resume, and as before without accumulation. A stand-in trainer whose
global_step advances where Lightning's optimizer step runs (inside the last micro-batch, before on_train_batch_end).
usage: python test_ema_accumulation.py"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.callbacks.simple_ema import SimpleEMA


class Trainer:
    global_step = 0


def run(start, accum, batches):
    t = Trainer(); t.global_step = start
    ema = SimpleEMA(decay=0.9996)
    calls = []
    ema.ema_step = lambda: calls.append(t.global_step)
    ema.on_train_start(t, None)
    for b in range(batches):
        if b % accum == accum - 1:
            t.global_step += 1
        ema.on_train_batch_end(t, None, None, None, b)
    return calls


ok = True
for start, accum, batches, want in ((0, 4, 40, list(range(1, 11))), (1000, 4, 8, [1001, 1002]),
                                    (0, 1, 5, [1, 2, 3, 4, 5]), (1000, 1, 3, [1001, 1002, 1003])):
    got = run(start, accum, batches)
    good = got == want
    ok &= good
    print(f"start {start}, accumulation {accum}, {batches} batches: EMA updates at steps {got}  {'ok' if good else 'FAIL'}")
print("EMA_ACCUM_TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
