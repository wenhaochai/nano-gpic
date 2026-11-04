#!/usr/bin/env python3
"""How much of a real model's sample variance comes from the noise? From jobs/eval_noise.sh's arms: two arms with one shared noise tensor each (fixed1, fixed2) and one arm with a noise per prompt (prompt_a).

fixed1[i] and fixed2[i] have the same prompt and different noise, so E ||f1 - f2||^2 / 2 is the noise's part of the variance within a prompt, in the features or in the pixels.
The total is the trace of the covariance over prompts with a noise per prompt. The official protocol removes the noise's part from the covariance and adds a shift of the mean (one tensor is not the average over tensors).
usage: noise_share_real.py <eval_noise output dir> <step>
"""
import json
import os
import sys

import numpy as np

out, step = sys.argv[1], sys.argv[2]
feat = lambda arm: np.load(f"{out}/{arm}_{step}/features.npy", mmap_mode="r").astype(np.float64)
f1, f2, fp = feat("fixed1"), feat("fixed2"), feat("prompt_a")
noise = ((f1 - f2) ** 2).sum(1).mean() / 2
total = np.trace(np.cov(fp, rowvar=False))
res = dict(step=step, feature_noise_var=noise, feature_total_var=total, feature_noise_share=noise / total, trace_cov_fixed1=np.trace(np.cov(f1, rowvar=False)), trace_cov_fixed2=np.trace(np.cov(f2, rowvar=False)), trace_cov_prompt=total)
res["mean_shift_fixed1_vs_prompt"] = ((f1.mean(0) - fp.mean(0)) ** 2).sum()
res["mean_shift_fixed2_vs_prompt"] = ((f2.mean(0) - fp.mean(0)) ** 2).sum()
res["mean_shift_fixed1_vs_fixed2"] = ((f1.mean(0) - f2.mean(0)) ** 2).sum()
if os.path.exists(f"{out}/prompt_b_{step}/features.npy"):
    fb = feat("prompt_b")
    res["mean_shift_prompt_a_vs_b"] = ((fp.mean(0) - fb.mean(0)) ** 2).sum()
npz = lambda arm: [os.path.join(r, "output.npz") for r, _, fs in os.walk(f"{out}/{arm}_{step}/pred") if "output.npz" in fs][0]
try:
    a, b = np.load(npz("fixed1"))["arr_0"], np.load(npz("fixed2"))["arr_0"]
    n = min(5000, len(a))
    x, y = a[:n].astype(np.float32) / 127.5 - 1, b[:n].astype(np.float32) / 127.5 - 1
    res.update(pixel_noise_var=float(((x - y) ** 2).mean() / 2), pixel_var_over_prompts=float(x.var(0).mean()), pixel_mean_abs_diff_255=float(np.abs(a[:n].astype(np.float32) - b[:n].astype(np.float32)).mean()))
except (IndexError, OSError) as e:
    print("no pixel part:", e)
res = {k: float(v) if not isinstance(v, str) else v for k, v in res.items()}
json.dump(res, open(f"{out}/noise_share.json", "w"), indent=1)
print(json.dumps(res, indent=1))
