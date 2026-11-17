"""Reference statistics at a resolution R in the format of data/gpic/reference_stats/test_stats.npz (dino_mu, dino_sigma, inception_mu,
inception_sigma, tar_idx, key, dino_embeddings, inception_embeddings), from the features of crop_reference_images.py's test_r<R>.npy:
DINOv2 from tools/fast_eval.py --save-features, Inception from tools/official_inception_sharded.py features (the shards in order).
Only the rows in rows.npy (images found in the test tars) enter; mu and sigma are float64 (numpy mean and cov, rowvar False).
The shipped test_stats.npz is not self-consistent: its mu and sigma are not the mean and covariance of its own embeddings (FD 7.50 for
DINOv2, 0.669 for Inception between the two, 2026-11-17), so they come from other images. A rebuilt file is therefore checked on its
embeddings (the same images, mean and covariance of each side: FD 0.00034 and 0.000022 at 256), not on the shipped mu and sigma.
usage: reference_stats.py build <dino.npy> <rows.npy> <out.npz> <inception shard 0.npy> ... <shard S-1.npy>
       reference_stats.py compare <a.npz> <b.npz> [--max-rel <r>] [--max-fd-shared <x>] [--b-self]
         per backend: the FD between a's and b's stored mu and sigma (gpic_eval's FD), the FD between the mean and covariance of the
         embeddings of the keys both hold (fd_shared), and the per-image relative L2 distance of those embeddings; --b-self adds the FD
         between b's stored mu and sigma and the mean and covariance of b's own embeddings. Exits 1 if a median distance exceeds r or an
         fd_shared exceeds x.
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


def stats(e):
    return e.mean(0), np.cov(e, rowvar=False)


def compare(a, b, max_rel=None, max_fd_shared=None, b_self=False):
    from gpic_eval.metrics.fd import compute_FD_with_stats
    A, B = np.load(a), np.load(b)
    ka, kb = A["key"].tolist(), B["key"].tolist()   # once: an NpzFile reads a member from the file at every access (13.5 ms for 50K keys)
    pos = {k: i for i, k in enumerate(kb)}
    ia = [i for i, k in enumerate(ka) if k in pos]
    ib = [pos[ka[i]] for i in ia]
    res, ok = {"a": a, "b": b, "shared_keys": len(ia), "keys_a": len(ka), "keys_b": len(kb)}, True
    for name in ("dino", "inception"):
        fd, fd_mu, fd_sigma = compute_FD_with_stats(A[f"{name}_mu"], B[f"{name}_mu"], A[f"{name}_sigma"], B[f"{name}_sigma"])
        eb_all = B[f"{name}_embeddings"].astype(np.float64)
        ea, eb = A[f"{name}_embeddings"][ia].astype(np.float64), eb_all[ib]
        rel = np.linalg.norm(ea - eb, axis=1) / np.linalg.norm(eb, axis=1)
        (ma, sa), (mb, sb) = stats(ea), stats(eb)
        fd_shared = compute_FD_with_stats(ma, mb, sa, sb)[0]
        r = {"fd": float(fd), "fd_mu": float(fd_mu), "fd_sigma": float(fd_sigma), "fd_shared": float(fd_shared),
             "rel_l2_median": float(np.median(rel)), "rel_l2_p99": float(np.quantile(rel, 0.99)), "rel_l2_max": float(rel.max())}
        if b_self:
            m, s = stats(eb_all)
            r["fd_b_stats_vs_b_embeddings"] = float(compute_FD_with_stats(B[f"{name}_mu"], m, B[f"{name}_sigma"], s)[0])
        res[name] = r
        ok &= (max_rel is None or np.median(rel) <= max_rel) and (max_fd_shared is None or fd_shared <= max_fd_shared)
    print(json.dumps(res, indent=1))
    return ok


def opt(name):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else None


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif cmd == "compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3], opt("--max-rel"), opt("--max-fd-shared"), "--b-self" in sys.argv) else 1)
    else:
        sys.exit(f"unknown command {cmd}")
