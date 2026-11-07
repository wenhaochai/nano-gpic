"""gpic-eval, sharded over GPUs: the same DINOv2 features and the same metric code, without the CLI's bottlenecks.

torchrun --nproc_per_node N tools/fast_eval.py <samples .npy|.npz (uint8 NHWC)> <reference stats .npz> <out.json>
         [--batch 256] [--workers 12]

What is identical to `gpic-eval <samples> <ref>`: the DINOv2 ViT-L/14 model (gpic_eval.backends.dino.build_dinov2, fp32),
the per-image preprocessing (_dino_transform: PIL bicubic resize to 224, ImageNet normalization), the feature extractor
(extract_features_from_npz) and the metrics (gpic_eval.eval.eval_from_embeddings: FD from the reference mu/sigma, PRDC
and MMD from the reference embeddings). What changes: each rank extracts features for a contiguous slice of the images
with DataLoader workers (the CLI defaults to num_workers=0 on one GPU), samples are read from a uint8 array (no PNG
round trip; PNG is lossless, so the pixels are the same), and rank 0 concatenates the features in image order.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset


class Slice(Dataset):
    def __init__(self, arr, lo, hi, transform):
        self.arr, self.lo, self.hi, self.t = arr, lo, hi, transform

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, i):
        return self.t(np.asarray(self.arr[self.lo + i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples"); ap.add_argument("ref"); ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from gpic_eval.backends.dino import build_dinov2, extract_features_from_npz, _dino_transform
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    t0 = time.time()
    arr = np.load(a.samples, mmap_mode="r") if a.samples.endswith(".npy") else np.load(a.samples)["arr_0"]
    n = arr.shape[0]
    lo, hi = rank * n // world, (rank + 1) * n // world
    loader = DataLoader(Slice(arr, lo, hi, _dino_transform), batch_size=a.batch, num_workers=a.workers)
    model = build_dinov2(dev)
    feats = extract_features_from_npz(loader, model, dev)
    t_feat = time.time() - t0
    obj = [None] * world
    dist.all_gather_object(obj, (lo, np.asarray(feats, dtype=np.float32)))
    if rank == 0:
        from gpic_eval.eval import eval_from_embeddings
        from gpic_eval.models import FDStatistics
        allf = np.concatenate([f for _, f in sorted(obj, key=lambda x: x[0])])
        ref = np.load(a.ref)
        stats = {"dino": FDStatistics(ref["dino_mu"], ref["dino_sigma"])} if "dino_mu" in ref.files else {}
        emb = {"dino": ref["dino_embeddings"]} if "dino_embeddings" in ref.files else {}
        np.random.seed(0)   # PRDC subsamples 10K features with np.random.choice; the CLI leaves it unseeded
        res = eval_from_embeddings({"dino": allf}, ref_embeddings=emb, ref_stats=stats)
        out = {k: v.get("dino") for k, v in vars(res).items() if isinstance(v, dict)}
        out.update({"n": int(allf.shape[0]), "world": world, "feature_s": t_feat, "total_s": time.time() - t0})
        json.dump(out, open(a.out, "w"), indent=1)
        print(json.dumps(out, indent=1))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
