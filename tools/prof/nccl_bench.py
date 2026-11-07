"""All-reduce bandwidth on this node at the job's gradient sizes (torchrun --nproc_per_node 8).

The job all-reduces ~4.49 GB of fp32 gradients per step in DDP buckets (default cap 25 MB). Measure, for fp32 and
bf16: one 25 MB bucket, the whole gradient as 25 MB buckets back to back, and one flat 4.49 GB tensor. Reports
algorithm bandwidth (bytes / time) and bus bandwidth (x 2(n-1)/n), the number to compare with NVLink.
"""
import json, os, statistics, sys

import torch
import torch.distributed as dist


def bench(fn, warm=3, iters=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main(out):
    dist.init_process_group("nccl")
    r, n = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    total = 1_122_003_712  # denoiser trainable parameters
    res = {"world": n}
    for dt, sz in ((torch.float32, 4), (torch.bfloat16, 2)):
        bucket = torch.ones(25 * 2**20 // sz, dtype=dt, device="cuda")
        nb = total * sz // (25 * 2**20) + 1
        flat = torch.ones(total, dtype=dt, device="cuda")
        t_one = bench(lambda: dist.all_reduce(bucket))
        t_seq = bench(lambda: [dist.all_reduce(bucket) for _ in range(nb)], 1, 5)
        t_flat = bench(lambda: dist.all_reduce(flat), 1, 5)
        f = 2 * (n - 1) / n
        for nm, t, by in (("bucket_25MB", t_one, bucket.numel() * sz), (f"{nb}x25MB", t_seq, nb * bucket.numel() * sz),
                          ("flat", t_flat, total * sz)):
            res[f"{str(dt).split('.')[-1]}_{nm}"] = {"ms": t, "GB": by / 1e9, "algbw_GBps": by / t / 1e6,
                                                     "busbw_GBps": by / t / 1e6 * f}
        del flat
        torch.cuda.empty_cache()
    if r == 0:
        json.dump(res, open(out, "w"), indent=1)
        print(json.dumps(res, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1])
