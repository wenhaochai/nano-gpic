#!/usr/bin/env python3
"""Host time of the per-step parameter work on DTensors (lever B) against plain tensors, on CPU (gloo, one rank): the
EMA's foreach lerp and AdamW's step over d1024's 129 sharded parameters, with tiny tensors so that only the dispatch
cost remains. Kernel time on the GPU is not measured here.
usage: CUDA_VISIBLE_DEVICES= python dtensor_overhead.py [n tensors] [elements per tensor]
"""
import os, sys, time
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard


def bench(fn, n=50):
    for _ in range(5):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 129
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group("gloo", rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    torch.set_num_threads(1)
    plain = [torch.randn(E) for _ in range(N)]
    plain2 = [torch.randn(E) for _ in range(N)]
    dt = [distribute_tensor(t.clone(), mesh, [Shard(0)]) for t in plain]
    dt2 = [distribute_tensor(t.clone(), mesh, [Shard(0)]) for t in plain2]
    loc, loc2 = [t.to_local() for t in dt], [t.to_local() for t in dt2]
    print(f"{N} tensors of {E} elements, one rank, CPU, ms per call")
    print(f"foreach_lerp_ plain   {bench(lambda: torch._foreach_lerp_(plain, plain2, 1e-4)):.3f}")
    print(f"foreach_lerp_ DTensor {bench(lambda: torch._foreach_lerp_(dt, dt2, 1e-4)):.3f}")
    print(f"foreach_lerp_ local   {bench(lambda: torch._foreach_lerp_(loc, loc2, 1e-4)):.3f}")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baselines", "PixelGen"))
    from src.utils.fsdp_setup import ShardLocalAdamW
    ps = [torch.nn.Parameter(distribute_tensor(t.clone(), mesh, [Shard(0)])) for t in plain]
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = ShardLocalAdamW(ps, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    print(f"AdamW fused   DTensor via ShardLocalAdamW {bench(opt.step, 20):.3f}")
    for kind in ("foreach", "fused"):
        for name, ps in (("plain", [torch.nn.Parameter(t.clone()) for t in plain]),
                         ("DTensor", [torch.nn.Parameter(distribute_tensor(t.clone(), mesh, [Shard(0)])) for t in plain])):
            for p in ps:
                p.grad = torch.randn_like(p)
            try:
                opt = torch.optim.AdamW(ps, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, **{kind: True})
                print(f"AdamW {kind:7s} {name:7s} {bench(opt.step, 20):.3f}")
            except Exception as e:
                print(f"AdamW {kind:7s} {name:7s} unsupported: {str(e)[:100]}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
