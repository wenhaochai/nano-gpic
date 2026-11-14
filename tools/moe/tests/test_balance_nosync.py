#!/usr/bin/env python3
"""The balance callback's update without host syncs (src/callbacks/moe_balance.py, 2026-11-14 after the step profile w2686) against
the version it replaces (copied below as ref_update): for routes tc and ecb, random statistics over 60 steps (some with no call),
every routing buffer is bit for bit equal after every step. On CUDA the new update also runs under
torch.cuda.set_sync_debug_mode("error"), which raises on any host sync.
usage: python test_balance_nosync.py   (CPU; a GPU adds the sync check)"""
import copy, os, sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.callbacks.moe_balance import MoEBalance          # noqa: E402
from src.models.layers.moe import MoEFeedForward          # noqa: E402

OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {detail}", flush=True)


@torch.no_grad()
def ref_update(cb):   # the update before 2026-11-14 (branches on GPU tensors), without the all_reduce
    stat = torch.stack([m.moe_stat for m in cb.layers])
    for m, s in zip(cb.layers, stat):
        n = m.n
        load, thr, calls = s[:n], s[n:2 * n], s[2 * n + 2]
        if calls > 0:
            if m.route == "tc":
                m.moe_bias.add_(m.bias_rate * torch.sign(load.mean() - load))
            elif m.route == "ecb":
                mean = thr / calls
                if bool(m.moe_thr_set):
                    m.moe_thr.lerp_(mean, 1 - cb.thr_decay)
                else:
                    m.moe_thr.copy_(mean)
                    m.moe_thr_set.fill_(True)
        m.moe_stat.zero_()
    return stat


def make(dev):
    torch.manual_seed(0)
    net = nn.ModuleList([MoEFeedForward(64, 256, experts=8, active=2, route=r, tokens=16) for r in ("tc", "ecb", "ecb", "tc", "eci")]).to(dev)
    cb = MoEBalance(log_every=10)
    cb.setup_layers(net)
    return net, cb


for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    net_a, cb_a = make(dev)
    net_b, cb_b = make(dev)
    g = torch.Generator().manual_seed(1)
    worst = 0
    for step in range(60):
        n = 8
        for ma, mb in zip(cb_a.layers, cb_b.layers):
            calls = 0.0 if step in (0, 7, 8, 31) else float(torch.randint(1, 4, (), generator=g))
            load = torch.randint(0, 50, (n,), generator=g).float() * (calls > 0)
            thr = torch.rand(n, generator=g) * calls
            v = torch.cat([load, thr, torch.tensor([float(torch.randint(0, 5, (), generator=g)), 512.0 * calls, calls])]).to(dev)
            ma.moe_stat.copy_(v); mb.moe_stat.copy_(v)
        if dev == "cuda" and step == 5:
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                cb_a.update()
                synced = False
            except RuntimeError as e:
                synced = True
                print("  sync:", str(e)[:120])
            torch.cuda.set_sync_debug_mode("default")
            check("cuda: the new update makes no host sync", not synced)
            # the control: the old update syncs, so the check can see a sync (on a copy, not on the compared layers)
            _, cb_c = make(dev)
            for mc, ma in zip(cb_c.layers, cb_a.layers):
                mc.moe_stat.fill_(1.0)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                ref_update(cb_c)
                ref_synced = False
            except RuntimeError:
                ref_synced = True
            torch.cuda.set_sync_debug_mode("default")
            check("cuda: the old update syncs (the check sees syncs)", ref_synced)
        else:
            cb_a.update()
        ref_update(cb_b)
        for ma, mb in zip(cb_a.layers, cb_b.layers):
            for name in ("moe_bias", "moe_thr", "moe_thr_set", "moe_stat"):
                a, b = getattr(ma, name), getattr(mb, name)
                if not torch.equal(a, b):
                    worst += 1
    check(f"{dev}: every buffer of 5 layers (tc, ecb, eci) equal to the old update after each of 60 steps", worst == 0, f"{worst} mismatches")
    check(f"{dev}: the ecb thresholds were set and moved", all(bool(m.moe_thr_set) and float((m.moe_thr - 0.5).abs().max()) > 0
                                                           for m in cb_a.layers if m.route == "ecb"))
    check(f"{dev}: the tc bias moved", all(float(m.moe_bias.abs().max()) > 0 for m in cb_a.layers if m.route == "tc"))
print("BALANCE_NOSYNC_TEST " + ("PASS" if OK else "FAIL")); sys.exit(0 if OK else 1)
