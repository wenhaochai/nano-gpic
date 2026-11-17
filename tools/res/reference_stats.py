"""Reference statistics at a resolution R in the format of data/gpic/reference_stats/test_stats.npz (dino_mu, dino_sigma, inception_mu,
inception_sigma, tar_idx, key, dino_embeddings, inception_embeddings), from the features of crop_reference_images.py's test_r<R>.npy:
DINOv2 from tools/fast_eval.py --save-features, Inception from tools/official_inception_sharded.py features (the shards in order).
Only the rows in rows.npy (images found in the test tars) enter; mu and sigma are float64 (numpy mean and cov, rowvar False).
usage: reference_stats.py build <dino.npy> <rows.npy> <out.npz> <inception shard 0.npy> ... <shard S-1.npy>
       reference_stats.py compare <a.npz> <b.npz> [--max-rel <r>]   FD between a and b per backend (gpic_eval's FD) and the per-image
       relative L2 distance of the embeddings of the keys both hold; with --max-rel, exits 1 if a median distance exceeds r
"""
import json, sys

import numpy as np

WC = "/scratch/gpfs/GROUP/USER"
SHIPPED = f"{WC}/data/gpic/reference_stats/test_stats.npz"


def build(dino, rows, out, shards):
    ref = np.load(SHIPPED)
    rows = np.load(rows)
    d = np.load(dino)
    inc = np.concatenate([np.load(s) for s in shards])
    n = len(ref["key"])
    assert d.shape == (n, 1024) and inc.shape == (n, 2048), f"features {d.shape} {inc.shape}, expected {n} rows"
    d, inc = d[rows].astype(np.float32), inc[rows].astype(np.float32)
    z = {"key": ref["key"][rows], "tar_idx": ref["tar_idx"][rows], "dino_embeddings": d, "inception_embeddings": inc}
    for name, f in (("dino", d), ("inception", inc)):
        f64 = f.astype(np.float64)
        z[f"{name}_mu"], z[f"{name}_sigma"] = f64.mean(0), np.cov(f64, rowvar=False)
    np.savez(out, **z)
    print(json.dumps({"out": out, "images": int(len(rows))}))


def compare(a, b, max_rel=None):
    from gpic_eval.metrics.fd import compute_FD_with_stats
    A, B = np.load(a), np.load(b)
    pos = {k: i for i, k in enumerate(B["key"].tolist())}
    ia = [i for i, k in enumerate(A["key"].tolist()) if k in pos]
    ib = [pos[A["key"][i]] for i in ia]
    res, ok = {"a": a, "b": b, "shared_keys": len(ia), "keys_a": int(len(A["key"])), "keys_b": int(len(B["key"]))}, True
    for name in ("dino", "inception"):
        fd, fd_mu, fd_sigma = compute_FD_with_stats(A[f"{name}_mu"], B[f"{name}_mu"], A[f"{name}_sigma"], B[f"{name}_sigma"])
        ea, eb = A[f"{name}_embeddings"][ia].astype(np.float64), B[f"{name}_embeddings"][ib].astype(np.float64)
        rel = np.linalg.norm(ea - eb, axis=1) / np.linalg.norm(eb, axis=1)
        res[name] = {"fd": float(fd), "fd_mu": float(fd_mu), "fd_sigma": float(fd_sigma),
                     "rel_l2_median": float(np.median(rel)), "rel_l2_max": float(rel.max())}
        ok &= max_rel is None or np.median(rel) <= max_rel
    print(json.dumps(res, indent=1))
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif cmd == "compare":
        mr = float(sys.argv[sys.argv.index("--max-rel") + 1]) if "--max-rel" in sys.argv else None
        sys.exit(0 if compare(sys.argv[2], sys.argv[3], mr) else 1)
    else:
        sys.exit(f"unknown command {cmd}")
