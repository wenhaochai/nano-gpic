#!/usr/bin/env python3
"""CPU test of MFU lever B2 (src/utils/fsdp_setup.ShardLocalAdamW and SimpleEMA's local lerp), 2 ranks over gloo: the same
parameters as Shard(0) DTensors (uneven first dims, a rank with an empty shard) plus a replicated plain group with its
own lr. torch.optim.AdamW(fused=True) on the DTensors against ShardLocalAdamW, 6 steps with the same gradients (one
parameter without a gradient at step 3), weight decay 0 and 0.1: parameters and states must be bitwise equal at every
step; the state_dicts alike; ShardLocalAdamW loaded from AdamW's state_dict continues bitwise equal; the EMA lerp on the
local tensors equals the lerp on the DTensors. usage: python test_local_opt.py"""
import os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
SHAPES = [(7, 5), (8, 3), (1, 4), (13,), (2, 2, 3), (64, 33)]


def run(rank, world, port):
    sys.path.insert(0, SRC)
    from src.utils.fsdp_setup import ShardLocalAdamW, _local
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard, DTensor
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(0)
    full = [torch.randn(s) for s in SHAPES]
    rep = [torch.randn(17), torch.randn(5)]
    loc = lambda t: t.to_local() if isinstance(t, DTensor) else t
    eq = lambda a, b: all(torch.equal(loc(x), loc(y)) for x, y in zip(a, b))
    ok = True
    for wd in (0.0, 0.1):
        def make():
            return ([torch.nn.Parameter(distribute_tensor(f.clone(), mesh, [Shard(0)])) for f in full],
                    [torch.nn.Parameter(r.clone()) for r in rep])
        (dA, pA), (dB, pB) = make(), make()
        kw = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd, fused=True)
        oA = torch.optim.AdamW([{"params": dA}, {"params": pA, "lr": 2e-3}], **kw)
        gB = [{"params": dB}, {"params": pB, "lr": 2e-3}]
        oB = ShardLocalAdamW.like(torch.optim.AdamW(gB, **kw), gB)

        def grads(step, ps, pl):
            g = torch.Generator().manual_seed(100 + step)
            for i, (p, f) in enumerate(zip(ps, full)):
                gf = torch.randn(f.shape, generator=g)
                p.grad = None if (step == 3 and i == 1) else distribute_tensor(gf, mesh, [Shard(0)])
            for p, r in zip(pl, rep):
                p.grad = torch.randn(r.shape, generator=g)

        for step in range(6):
            grads(step, dA, pA); grads(step, dB, pB)
            oA.step(); oB.step()
            same = eq(dA, dB) and eq(pA, pB) and all(
                eq([oA.state[a][k] for a in dA + pA if a in oA.state], [oB.state[b][k] for b in dB + pB if b in oB.state])
                for k in ("exp_avg", "exp_avg_sq", "step"))
            ok &= same
            if rank == 0:
                print(f"wd {wd} step {step}: parameters and states {'bitwise equal' if same else 'DIFFER'}")
        sA, sB = oA.state_dict(), oB.state_dict()
        alike = sA["param_groups"] == sB["param_groups"] and sA["state"].keys() == sB["state"].keys() and all(
            eq([sA["state"][i][k]], [sB["state"][i][k]]) for i in sA["state"] for k in sA["state"][i])
        # continue from AdamW's state in a fresh ShardLocalAdamW: one more step must equal AdamW's
        (dC, pC) = ([torch.nn.Parameter(distribute_tensor(p.detach().full_tensor().clone(), mesh, [Shard(0)])) for p in dA],
                    [torch.nn.Parameter(p.detach().clone()) for p in pA])
        gC = [{"params": dC}, {"params": pC, "lr": 2e-3}]
        oC = ShardLocalAdamW.like(torch.optim.AdamW(gC, **kw), gC)
        # load copies, as from a checkpoint (load_state_dict keeps the given tensors, so A and C would share their states)
        oC.load_state_dict({"state": {i: {k: v.clone() if torch.is_tensor(v) else v for k, v in st.items()}
                                      for i, st in sA["state"].items()},
                            "param_groups": [dict(g) for g in sA["param_groups"]]})
        grads(6, dA, pA); grads(6, dC, pC)
        oA.step(); oC.step()
        cont = eq(dA, dC) and eq(pA, pC)
        ok &= alike and cont
        if rank == 0:
            print(f"wd {wd}: state_dicts alike {alike}; ShardLocalAdamW from AdamW's state_dict, one more step equal {cont}")
    eA = [torch.nn.Parameter(distribute_tensor(f.clone() * 0.5, mesh, [Shard(0)])) for f in full]
    eB = [torch.nn.Parameter(distribute_tensor(f.clone() * 0.5, mesh, [Shard(0)])) for f in full]
    with torch.no_grad():
        for _ in range(3):
            torch._foreach_lerp_(eA, dA, 1e-4)
            torch._foreach_lerp_(_local(eB), _local(dA), 1e-4)
    ema = eq(eA, eB)
    ok &= ema
    if rank == 0:
        print(f"EMA lerp on local tensors = on DTensors: {ema}")
    t = torch.tensor([int(ok)])
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("ok" if t.item() else "FAIL")
    dist.destroy_process_group()
    if not t.item():
        raise SystemExit(1)


if __name__ == "__main__":
    mp.spawn(run, args=(2, 29541), nprocs=2, join=True)
