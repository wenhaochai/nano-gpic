#!/usr/bin/env python3
"""The official GPIC evaluation's Inception metrics for one set of samples (owner 2026-11-07: every evaluation records held-out loss, FD and every
other metric the official evaluation gives). gpic_eval.eval_with_ref_stats with models=["inception"]: TF Inception-v3 features of the samples
(the toolkit's classify_image_graph_def.pb), the reference's precomputed inception_mu/sigma/embeddings from test_stats.npz; FD, PRDC (k=5, 10K
subsample, the toolkit's defaults) and MMD. The DINOv2 metrics come from tools/fast_eval.py as before.
usage: official_inception.py <samples.npz> <ref_stats.npz> <out.json>   (env: envs/gpic-eval-tf; XDG_CACHE_HOME holds gpic_eval/classify_image_graph_def.pb)"""
import json, sys, time
from gpic_eval.eval import eval_with_ref_stats
sample, ref, out = sys.argv[1:4]
t0 = time.time()
r = eval_with_ref_stats(sample_path=sample, ref_path=ref, models=["inception"], batch_size=500)
res = {}
for f in ("fd", "fd_mu", "fd_sigma", "precision", "recall", "density", "coverage", "mmd"):
    v = getattr(r, f).get("inception")
    if v is not None:
        res[f"inception_{f}"] = float(v)
res["inception_s"] = round(time.time() - t0, 1)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
