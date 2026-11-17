"""FD of an evaluation's saved sample features (eval_gmflow.sh EVAL_SAVE_FEATURES=1: features/dino.npy and features/inception_shard<s>.npy)
against the mu and sigma of any reference statistics file (the shipped test_stats.npz or res/test_stats_r<R>.npz), on CPU: a re-scoring
needs no new sampling. FD only (gpic_eval's compute_FD_with_stats on float64 means and covariances); PRDC and MMD need the toolkit's full path.
usage: rescore_features.py <eval dir> <reference.npz> [<reference.npz> ...]
"""
import glob, json, os, re, sys

import numpy as np


def main():
    from gpic_eval.metrics.fd import compute_FD_with_stats
    d = sys.argv[1]
    dino = np.load(f"{d}/features/dino.npy")
    shards = sorted(glob.glob(f"{d}/features/inception_shard*.npy"), key=lambda p: int(re.search(r"shard(\d+)", p).group(1)))
    inc = np.concatenate([np.load(p) for p in shards]) if shards else None
    out = {"eval_dir": d, "dino_rows": int(len(dino)), "inception_rows": int(len(inc)) if inc is not None else 0, "scores": {}}
    feats = {"dino": dino.astype(np.float64)}
    if inc is not None:
        feats["inception"] = inc.astype(np.float64)
    stats = {k: (v.mean(0), np.cov(v, rowvar=False)) for k, v in feats.items()}
    for ref in sys.argv[2:]:
        R = np.load(ref)
        out["scores"][os.path.basename(ref)] = {k: float(compute_FD_with_stats(m, R[f"{k}_mu"], s, R[f"{k}_sigma"])[0]) for k, (m, s) in stats.items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
