#!/usr/bin/env python3
"""Is NCCL's all_gather_into_tensor placing each rank's data at its rank's block? (gpic-28 2026-11-08: in the d1536 ladder
an out-of-place fp32 all-gather of final_layer.linear.weight's shards, 96 x 1536 = 147456 elements per rank, placed rank
1's data at block 2, rank 2's at 3, rank 3's at 1 and swapped 5 and 6, while FSDP2's in-place bf16 all-gather of the same
data was right.) For per-rank element counts around that size and the other ladder widths' final layers, fp32 and bf16,
out-of-place and in-place: every rank checks which rank's data each block of its output holds; rank 0 prints every case
where any rank sees a block from the wrong rank. Plain torch.distributed, no model code.
usage: torchrun --standalone --nproc_per_node 8 nccl_ag_test.py"""
import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
r, W = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda")
sizes = sorted({24576, 49152, 73728, 98304, 96 * 1024, 96 * 1280, 140000, 147456, 150000, 96 * 1792, 196608, 245760, 294912,
                393216, 589824, 1179648, 96 * 2048})
bad = []
for n in sizes:
    for dt in (torch.float32, torch.bfloat16):
        for mode in ("out-of-place", "in-place"):
            for rep in range(3):
                out = torch.full((W * n,), -1.0, dtype=dt, device=dev)
                mine = torch.full((n,), float(r + 1), dtype=dt, device=dev)
                if mode == "in-place":
                    out[r * n:(r + 1) * n].copy_(mine)
                    dist.all_gather_into_tensor(out, out[r * n:(r + 1) * n])
                else:
                    dist.all_gather_into_tensor(out, mine)
                torch.cuda.synchronize()
                blk = out.view(W, n)
                src = [int(blk[j, 0].item()) - 1 if torch.all(blk[j] == blk[j, 0]) else -2 for j in range(W)]
                allr = [None] * W
                dist.all_gather_object(allr, src)
                if r == 0 and any(s != list(range(W)) for s in allr):
                    bad.append(f"n {n} ({n * torch.empty((), dtype=dt).element_size() / 1024:.0f} KB) {str(dt)[6:]} {mode} rep {rep}: "
                               + "; ".join(f"rank {q} blocks from {s}" for q, s in enumerate(allr) if s != list(range(W))))
if r == 0:
    env = {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "CUDA_VISIBLE"))}
    env["libnccl"] = sorted({l.split()[-1] for l in open("/proc/self/maps") if "libnccl" in l})   # the library actually loaded
    print(f"NCCL {torch.cuda.nccl.version()}, env {env}, host {os.uname().nodename}: {len(bad)} bad cases of "
          f"{len(sizes) * 2 * 2 * 3}", flush=True)
    for b in bad[:30]:
        print("  " + b, flush=True)
    print("NCCL_AG_TEST DONE", flush=True)
dist.destroy_process_group()
