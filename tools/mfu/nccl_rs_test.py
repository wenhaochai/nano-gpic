#!/usr/bin/env python3
"""Does NCCL's reduce_scatter_tensor give each rank its own reduced chunk, and all_reduce the right sum? (gpic-28 2026-11-08:
NCCL 2.26.2's single-node NVLS AllGather puts ranks' blocks out of order at some sizes, NVIDIA/nccl #1906; the upstream fix
also touches ReduceScatter, which FSDP2 runs on every unit's gradients.) Sizes: every FSDP2 unit of the ladder's d1280 (G 7),
d1536 and d1792 (G 1), per rank as FSDP2 pads them (dim 0 rounded up to a multiple of the world size), plus a sweep;
fp32 and bf16. reduce-scatter: every rank fills chunk j with j + 1, so rank j must receive 8 (j + 1); all-reduce: rank r
fills r + 1, the sum must be 36 everywhere. Rank 0 prints every wrong case. Plain torch.distributed.
usage: torchrun --standalone --nproc_per_node 8 nccl_rs_test.py <code worktree>"""
import math
import os
import sys

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
r, W = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda")
sys.path.insert(0, os.path.join(sys.argv[1], "baselines", "PixelGen"))
from src.models.transformer.JiT_T2I import JiT_T2I  # noqa: E402


def per_rank(params):
    return sum(math.ceil(p.shape[0] / W) * (p.numel() // p.shape[0]) if p.ndim else 1 for p in params)


units = {}
for d, nb, nt, g in ((1280, 14, 3, 7), (1536, 16, 4, 1), (1792, 18, 4, 1)):
    with torch.device("meta"):
        net = JiT_T2I(input_size=256, in_channels=3, hidden_size=d, num_groups=d // 64, num_blocks=nb, num_text_blocks=nt,
                      patch_size=16, txt_embed_dim=2048, txt_max_length=320)
    keep = {p for m in net.modules() if type(m).__name__ == "RMSNorm" for p in m.parameters()} | {net.pos_embed}
    for name, blocks in (("text", list(net.text_refine_blocks)), ("image", list(net.blocks))):
        for i in range(0, len(blocks), g):
            ps = [p for b in blocks[i:i + g] for p in b.parameters() if p not in keep]
            units.setdefault(per_rank(ps), f"d{d} {name} unit {i // g}")
    inner = {p for b in list(net.text_refine_blocks) + list(net.blocks) for p in b.parameters()}
    units.setdefault(per_rank([p for p in net.parameters() if p not in keep and p not in inner and p.requires_grad]), f"d{d} root unit")
sweep = [24576, 49152, 98304, 147456, 196608, 294912, 589824, 1179648, 2359296, 4718592]
cases = sorted(set(units) | set(sweep))
bad = []
for n in cases:
    for dt in (torch.float32, torch.bfloat16):
        for rep in range(2):
            inp = torch.arange(1, W + 1, dtype=dt, device=dev).repeat_interleave(n)   # chunk j holds j + 1
            out = torch.empty(n, dtype=dt, device=dev)
            dist.reduce_scatter_tensor(out, inp)
            torch.cuda.synchronize()
            got = int(out[0].item() // W) - 1 if torch.all(out == out[0]) else -2
            ar = torch.full((n,), float(r + 1), dtype=dt, device=dev)
            dist.all_reduce(ar)
            torch.cuda.synchronize()
            ar_ok = bool(torch.all(ar == W * (W + 1) / 2))
            res = [None] * W
            dist.all_gather_object(res, (got, ar_ok))
            if r == 0:
                rs_bad = [(q, g_) for q, (g_, _) in enumerate(res) if g_ != q]
                ar_bad = [q for q, (_, ok) in enumerate(res) if not ok]
                if rs_bad or ar_bad:
                    bad.append(f"n {n} ({n * torch.empty((), dtype=dt).element_size() / 1024:.0f} KB, {units.get(n, 'sweep')}) "
                               f"{str(dt)[6:]} rep {rep}: reduce-scatter rank->chunk {rs_bad or 'ok'}; all-reduce wrong on {ar_bad or 'none'}")
if r == 0:
    env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    print(f"NCCL {torch.cuda.nccl.version()}, env {env}, host {os.uname().nodename}: units {sorted(units.items())}", flush=True)
    print(f"{len(bad)} bad cases of {len(cases) * 2 * 2}", flush=True)
    for b in bad[:40]:
        print("  " + b, flush=True)
    print("NCCL_RS_TEST DONE", flush=True)
dist.destroy_process_group()
