#!/usr/bin/env python3
"""CPU test (2 ranks, gloo) of src/utils/fsdp_setup.shard with 1 and 3 blocks per FSDP2 unit (GPIC_MFU_FSDP_GROUP) and of
ShardLocalAdamW in a real FSDP2 step: a small model with text and main blocks, RMSNorm weights replicated, a fixed
pos_embed; 3 steps (forward, backward, replicated-gradient sync, clip, AdamW) on rank-specific data. The full parameters
and gradients at every step: 1-block units with torch's AdamW = 3-block units with ShardLocalAdamW, bitwise. Against a plain
data-parallel reference (full model on every rank, gradients averaged): the first step's gradients within 1e-6 of each
tensor's largest value (a wrong scale, a missing sync or a misplaced shard shows here; AdamW would hide a scale), the
parameters within 1e-3 (Adam's g / (|g| + eps) turns rounding-level gradient differences of tiny gradients into larger
parameter differences). usage: python test_fsdp_group_cpu.py"""
import os, sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))


def run(rank, world, port):
    sys.path.insert(0, SRC)
    from src.models.layers.rmsnorm import RMSNorm
    from src.utils.fsdp_setup import shard, sync_replicated_grads, clip_grad_norm_, ShardLocalAdamW
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    _ar = dist.all_reduce   # gloo has no ReduceOp.AVG (NCCL, the production backend, has): SUM, then divide

    def all_reduce(t, op=dist.ReduceOp.SUM, **kw):
        if op == dist.ReduceOp.AVG:
            r = _ar(t, op=dist.ReduceOp.SUM, **kw)
            t.div_(dist.get_world_size())
            return r
        return _ar(t, op=op, **kw)
    dist.all_reduce = all_reduce
    _rs = dist.reduce_scatter_tensor   # FSDP2 reduces fp32 gradients with ReduceOp.AVG, which gloo sums: SUM, then divide

    def reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, **kw):
        if op == dist.ReduceOp.AVG:
            assert not kw.get("async_op")
            r = _rs(output, input, op=dist.ReduceOp.SUM, **kw)
            output.div_(dist.get_world_size())
            return r
        return _rs(output, input, op=op, **kw)
    dist.reduce_scatter_tensor = reduce_scatter_tensor

    class Blk(nn.Module):
        def __init__(s, d):
            super().__init__(); s.norm1 = RMSNorm(d); s.fc = nn.Linear(d, 3 * d + 1); s.out = nn.Linear(3 * d + 1, d)
        def forward(s, x):
            return x + s.out(torch.nn.functional.silu(s.fc(s.norm1(x))))

    class Net(nn.Module):
        def __init__(s, d=12):
            super().__init__()
            s.pos_embed = nn.Parameter(torch.randn(1, 5, d), requires_grad=False)
            s.text_refine_blocks = nn.ModuleList([Blk(d) for _ in range(2)])
            s.blocks = nn.ModuleList([Blk(d) for _ in range(5)])
            s.head = nn.Linear(d, 3)
        def forward(s, x):
            x = x + s.pos_embed
            for b in list(s.text_refine_blocks) + list(s.blocks):
                x = b(x)
            return s.head(x).pow(2).mean()

    def data(step):
        g = torch.Generator().manual_seed(1000 * step + rank)
        return torch.randn(4, 5, 12, generator=g)

    def train(mode):
        torch.manual_seed(0)
        net = Net()
        kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
        if mode == "ref":
            ps = [p for p in net.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(ps, **kw)
        else:
            group, local = (1, False) if mode == "g1" else (3, True)
            rep = shard(net, mesh, compute_bf16=False, compile_blocks=False, group=group)
            ps = [p for p in net.parameters() if p.requires_grad]
            groups = [{"params": [p for p in ps if isinstance(p, DTensor)]}, {"params": [p for p in ps if not isinstance(p, DTensor)]}]
            opt = (ShardLocalAdamW.like(torch.optim.AdamW(groups, **kw), groups) if local else torch.optim.AdamW(groups, **kw))
        out = []
        for step in range(3):
            loss = net(data(step))
            loss.backward()
            if mode == "ref":
                for p in ps:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(ps, 0.05)
            else:
                sync_replicated_grads(rep)
                clip_grad_norm_(ps, 0.05)
            with torch.no_grad():
                full = lambda t: t.full_tensor() if isinstance(t, DTensor) else t
                grads = [full(p.grad).detach().clone() for p in ps]
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                out.append(([full(p).detach().clone() for p in net.parameters()], grads))
        return out

    ref, g1, g3 = train("ref"), train("g1"), train("g3")
    ok = True
    rel = lambda xs, ys: max(float((a - b).abs().max() / b.abs().max().clamp_min(1e-12)) for a, b in zip(xs, ys))
    if os.environ.get("TEST_VERBOSE") and rank == 0:
        torch.manual_seed(0)
        names = [n for n, p in Net().named_parameters() if p.requires_grad]
        for n, a, b in zip(names, g1[0][1], ref[0][1]):
            r = float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))
            print(f"  {n:28s} rel {r:.1e}  |g| fsdp {float(a.norm()):.3e} ref {float(b.norm()):.3e}")
    for step in range(3):
        bit = all(torch.equal(a, b) for a, b in zip(g1[step][0] + g1[step][1], g3[step][0] + g3[step][1]))
        perr, gerr = rel(g1[step][0], ref[step][0]), rel(g1[step][1], ref[step][1])
        ok &= bit and perr < 1e-3 and (step > 0 or gerr < 1e-6)
        if rank == 0:
            print(f"step {step}: 1-block units + AdamW = 3-block units + ShardLocalAdamW, parameters and gradients bitwise {bit}; "
                  f"against the data-parallel reference: gradients {gerr:.1e}, parameters {perr:.1e}")
    t = torch.tensor([int(ok)])
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("ok" if t.item() else "FAIL")
    dist.destroy_process_group()
    if not t.item():
        raise SystemExit(1)


if __name__ == "__main__":
    mp.spawn(run, args=(2, 29547), nprocs=2, join=True)
