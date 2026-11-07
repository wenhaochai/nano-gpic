#!/usr/bin/env python3
"""CPU tests of MFU lever B3 (src/utils/fsdp_setup.py, GPIC_MFU_FSDP_FLAT=1: one flat parameter per FSDP2 unit).
1. 2 ranks, gloo, a small model with text and main blocks, RMSNorm weights replicated, a fixed pos_embed: 3 steps
   (forward, backward, replicated sync, clip, ShardLocalAdamW) with 3-block units, flat against per-parameter: the full
   parameters (the flat ones split back by the key map) and gradients equal bit for bit after step 0, and within 1e-6
   after steps 1-2 (CPU's fused AdamW rounds tensor ends differently; see below); with an acting clip within 1e-5.
2. One process: a block whose parameters are views of a flat parameter, compiled (Inductor, CPU): forward and
   backward equal to the same block without views in eager mode, and no recompilation over FSDP2-like swaps (the
   registered parameter alternates between a sharded stand-in and one persistent unsharded parameter).
3. The checkpoint key map: split_flat gives the original keys and tensors; _FlatView rebuilds the flat tensor.
usage: python test_fsdp_flat_cpu.py"""
import copy, os, sys
import torch
import torch._dynamo
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
sys.path.insert(0, SRC)


def make_net(d=12):
    from src.models.layers.rmsnorm import RMSNorm

    class Blk(nn.Module):
        def __init__(s, d):
            super().__init__(); s.norm1 = RMSNorm(d); s.fc = nn.Linear(d, 3 * d + 1); s.out = nn.Linear(3 * d + 1, d)
        def forward(s, x):
            return x + s.out(torch.nn.functional.silu(s.fc(s.norm1(x))))

    class Net(nn.Module):
        def __init__(s):
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
    return Net


def run(rank, world, port):
    from src.utils.fsdp_setup import shard, sync_replicated_grads, clip_grad_norm_, ShardLocalAdamW, split_flat
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    _ar, _rs = dist.all_reduce, dist.reduce_scatter_tensor   # gloo has no ReduceOp.AVG: SUM, then divide (NCCL averages)

    def all_reduce(t, op=dist.ReduceOp.SUM, **kw):
        if op == dist.ReduceOp.AVG:
            r = _ar(t, op=dist.ReduceOp.SUM, **kw); t.div_(dist.get_world_size()); return r
        return _ar(t, op=op, **kw)

    def reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, **kw):
        if op == dist.ReduceOp.AVG:
            r = _rs(output, input, op=dist.ReduceOp.SUM, **kw); output.div_(dist.get_world_size()); return r
        return _rs(output, input, op=op, **kw)
    dist.all_reduce, dist.reduce_scatter_tensor = all_reduce, reduce_scatter_tensor
    Net = make_net()

    def data(step):
        g = torch.Generator().manual_seed(1000 * step + rank)
        return torch.randn(4, 5, 12, generator=g)

    def train(flat, max_norm):
        torch.manual_seed(0)
        net = Net()
        ref_keys = [n for n, p in net.named_parameters()]
        if flat:
            os.environ["GPIC_MFU_FSDP_FLAT"] = "1"
        rep = shard(net, mesh, compute_bf16=False, compile_blocks=False, group=3)
        os.environ.pop("GPIC_MFU_FSDP_FLAT", None)
        kmap = net.__dict__.get("_b3_kmap", [])
        ps = [p for p in net.parameters() if p.requires_grad]
        groups = [{"params": [p for p in ps if isinstance(p, DTensor)]}, {"params": [p for p in ps if not isinstance(p, DTensor)]}]
        kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
        opt = ShardLocalAdamW.like(torch.optim.AdamW(groups, **kw), groups)
        full = lambda t: t.full_tensor() if isinstance(t, DTensor) else t
        out = []
        for step in range(3):
            loss = net(data(step))
            loss.backward()
            sync_replicated_grads(rep)
            clip_grad_norm_(ps, max_norm)
            with torch.no_grad():
                grads = split_flat({n: full(p.grad).detach().clone() for n, p in net.named_parameters() if p.grad is not None}, kmap)
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                params = split_flat({n: full(p).detach().clone() for n, p in net.named_parameters()}, kmap)
            out.append((params, grads, float(loss)))
        return out, ref_keys, kmap

    # a clip that does not act (coefficient exactly 1, as in production, where the grad norm is ~0.2 against 1.0): the
    # flat units must give the same bits; a clip that acts: the global norm sums its per-tensor norms in another grouping
    # (17 flat tensors against ~100), so the coefficient may differ in its last bits, and with it everything after
    a, keys, _ = train(False, 1e9)
    b, _, kmap = train(True, 1e9)
    ok = len(kmap) > 0
    for step in range(3):
        pa, ga, la = a[step]
        pb, gb, lb = b[step]
        same = (set(pa) == set(pb) == set(keys) and all(torch.equal(pa[k], pb[k]) for k in pa)
                and set(ga) == set(gb) and all(torch.equal(ga[k], gb[k]) for k in ga) and la == lb)
        # from step 1 on, CPU's fused AdamW rounds a few elements near the end of a tensor differently (1 ulp): the
        # per-parameter shards end where the flat shard does not. The step-0 bits (forward, backward, reduce-scatter,
        # first update) carry the claim; later steps must stay at the rounding level. (CUDA's fused AdamW does the same
        # arithmetic on every element: tools/mfu/tests/test_fused_adamw_ema.py, odd sizes included.)
        rel = max(float((pa[k] - pb[k]).abs().max() / pa[k].abs().max().clamp_min(1e-12)) for k in pa)
        ok &= same if step == 0 else (set(pa) == set(pb) and rel < 1e-6 and la == lb)
        if rank == 0:
            print(f"step {step}, no clip: flat = per-parameter FSDP2 (3-block units), loss, every parameter and gradient "
                  f"bitwise: {same}; largest relative parameter difference {rel:.1e}")
            if not same and os.environ.get("TEST_VERBOSE"):
                print(f"   loss {la!r} vs {lb!r}")
                for k in sorted(ga):
                    if not torch.equal(ga[k], gb[k]):
                        print(f"   grad {k}: max |diff| {float((ga[k] - gb[k]).abs().max()):.2e} of {float(ga[k].abs().max()):.2e}")
                for k in sorted(pa):
                    if not torch.equal(pa[k], pb[k]):
                        idx = (pa[k] != pb[k]).nonzero().tolist()
                        print(f"   param {k} {tuple(pa[k].shape)}: max |diff| {float((pa[k] - pb[k]).abs().max()):.2e} at {idx[:6]}")
    a, _, _ = train(False, 0.05)
    b, _, _ = train(True, 0.05)
    rel = max(float((a[s][0][k] - b[s][0][k]).abs().max() / a[s][0][k].abs().max().clamp_min(1e-12)) for s in range(3) for k in a[s][0])
    ok &= rel < 1e-5
    if rank == 0:
        print(f"clip acting (max_norm 0.05): largest relative parameter difference over 3 steps {rel:.1e}")
    t = torch.tensor([int(ok)])
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("distributed: " + ("ok" if t.item() else "FAIL"))
    dist.destroy_process_group()
    if not t.item():
        raise SystemExit(1)


def single_process_checks():
    from src.utils.fsdp_setup import flatten_unit, split_flat, _FlatView, FLAT
    from src.models.layers.rmsnorm import RMSNorm
    Net = make_net()
    torch.manual_seed(0)
    net = Net()
    ref = copy.deepcopy(net)
    keep = {p for m in net.modules() if isinstance(m, RMSNorm) for p in m.parameters()} | {net.pos_embed}
    names = {m: n for n, m in net.named_modules()}
    kmap = flatten_unit([net.blocks[0], net.blocks[1]], keep, names) + flatten_unit([net.blocks[2]], keep, names)
    ok = True
    # 3. key map: split_flat and _FlatView
    sd = split_flat(net.state_dict(), kmap)
    rsd = ref.state_dict()
    km = set(sd) == set(rsd) and all(torch.equal(sd[k], rsd[k]) for k in rsd)
    fv = _FlatView(rsd, kmap)
    km &= all(torch.equal(fv[f"blocks.{i}.{FLAT}"], getattr(net.blocks[i], FLAT).detach()) for i in (0, 2))
    km &= ("blocks.0.fc.weight" in fv) and (f"blocks.0.{FLAT}" in fv) and (f"blocks.1.{FLAT}" not in fv)
    print(f"key map: split_flat = the original state dict, _FlatView rebuilds the flat tensors: {km}")
    ok &= km
    # 2. compiled block with views: equal to eager without views, no recompile when the flat parameter object changes

    torch._dynamo.reset()
    blk, rblk = net.blocks[0], ref.blocks[0]
    cblk = torch.compile(blk)
    x = torch.randn(4, 5, 12, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    y, yr = cblk(x), rblk(xr)
    y.sum().backward(); yr.sum().backward()
    flat = getattr(blk, FLAT)
    gref = torch.cat([p.grad.reshape(-1) for n, p in rblk.named_parameters() if p.requires_grad and p not in
                      {rblk.norm1.weight}])
    eq = torch.allclose(y, yr, rtol=1e-6, atol=1e-6) and torch.allclose(x.grad, xr.grad, rtol=1e-6, atol=1e-6)
    # the flat gradient covers blocks 0 and 1; block 0's part comes first
    eq &= torch.allclose(flat.grad[:gref.numel()], gref, rtol=1e-6, atol=1e-6)
    from torch._dynamo.utils import counters
    # FSDP2 swaps the registered parameter between two fixed objects every step: the sharded DTensor between steps and
    # the persistent unsharded parameter during the unit's forward and backward (FSDPParam._unsharded_param)
    unsharded = nn.Parameter(flat.detach().clone())
    blk._parameters[FLAT] = unsharded
    cblk(torch.randn(4, 5, 12)).sum().backward()   # FSDP2's first step: compiled with the unsharded object in place
    n0 = counters["stats"]["unique_graphs"]
    for _ in range(3):
        blk._parameters[FLAT] = nn.Parameter(torch.zeros(1))   # the sharded stand-in between steps (never read)
        blk._parameters[FLAT] = unsharded                       # back for the forward
        cblk(torch.randn(4, 5, 12)).sum().backward()
    rec = counters["stats"]["unique_graphs"] - n0
    print(f"compiled block with views: forward and gradients equal to eager without views {eq}; new graphs over 3 "
          f"FSDP2-like swaps after the first compile: {rec}")
    ok &= eq and rec == 0
    return ok


if __name__ == "__main__":
    ok = single_process_checks()
    mp.spawn(run, args=(2, 29561), nprocs=2, join=True)
    print("FLAT_TEST " + ("PASS" if ok else "FAIL (single-process checks)"))
    sys.exit(0 if ok else 1)
